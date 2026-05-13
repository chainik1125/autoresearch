"""PREPARE workflow — pre-flight review + auto-apply mechanical patches.

Runs before the actual compute dispatch. Reads the cloned project repo,
the target Pipeline class, and the user's intent. Identifies blockers,
categorizes each into MECHANICAL (auto-applied) or USER-INPUT-NEEDED
(flag-and-abort). Commits applied patches to a new branch and pushes;
transfer's prep gate (in workflows/transfer.py) checks out that branch
before running.

### What v0 does

  1. Single-shot LLM call. Asks Claude to emit a structured JSON plan:
     `{verdict, rationale, patches: [{file, find, replace, reason}],
       user_input_blockers: [...]}`
  2. If `verdict == "user_input_needed"` or any blocker present: write
     LOUD [USER-INPUT-NEEDED] findings and exit (no patches applied).
     Transfer's gate sees the [USER-INPUT-NEEDED] tag and aborts.
  3. If `verdict == "patches_applied"`: deterministic Python applies
     each `find`/`replace` to the named file (verbatim text match). If
     any patch's `find` text isn't found in the file, abort patching —
     don't apply partial sets. Then `git checkout -b
     autoresearch/prepared-<run_id> && git commit && git push`.
  4. If `verdict == "ok"`: no patches, no branch, no findings.

The patch application is deterministic Python, not LLM tool-use. That
means every change is auditable from the agent's JSON output + the git
diff. Trade-off: the agent has to express patches as exact-text find/
replace, which is less flexible than Edit-tool calls but much easier to
audit. v1 trajectory: switch to Claude Agent SDK with the Edit tool
for cases where verbatim find/replace isn't enough.

### What's still v1

  - Multi-file refactors that don't fit verbatim find/replace
  - Adding new files (today: edit-existing only)
  - System-level changes (apt install, system symlinks) — though we
    could special-case symlink instructions since the Arditi case
    showed they're common

### Concrete prior example

  Arditi's `run_from_config.py` references
  `/root/git/dictionary_learning/data/misaligned_aggregated.jsonl`.
  Our pod clones to `/workspace/arditi_dl/...`. A future project that
  hardcodes `/root/git/<repo>/...` should yield a [MECHANICAL] patch
  that either edits the path OR adds an `os.symlink` call in the
  appropriate setup file. (The Arditi case itself is already handled
  by the fra_proj wrapper's `_ensure_hardcoded_symlink()` — that's a
  wrapper-level fix, not a code edit. The prep agent should recognize
  the SHAPE of the problem and propose the appropriate fix.)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.core import findings as findings_mod
from autoresearch.core.agent_runner import run_agent
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run, RunStatus


PREP_SYSTEM = """You are a research-engineering reviewer doing pre-flight static
analysis of a project that's about to be dispatched onto a GPU pod, off the
researcher's laptop. They're going to disengage as soon as the dispatch fires.

YOUR DEFAULT IS "no changes needed." Most projects are fine as-is — don't
invent issues to look helpful.

When you DO find something, categorize it:

  - **MECHANICAL** — can be fixed by exact-text find-and-replace in a file
    without asking the user. The deterministic post-step will apply each
    patch verbatim. Patches must be specific enough that the `find` text
    appears EXACTLY ONCE in the named file. Examples:
      * Hardcoded absolute paths in source code (`/root/...`, `/home/...`,
        `/Users/...`, `/tmp/<user-specific>`). Patch them to use env vars
        or relative-to-workspace paths.
      * Missing/wrong torch+torchvision pin (we hit this on Qwen-32B —
        torch was upgraded but torchvision wasn't, ABI mismatch). Patch
        requirements.txt to also pin the matching torchvision.
      * Missing apt/pip deps that imports will choke on. Patch the
        requirements.txt.

    Concrete worked example we've handled: Arditi's `run_from_config.py`
    references `/root/git/dictionary_learning/data/misaligned_aggregated.jsonl`.
    Their clone lives at `/workspace/arditi_dl/...` on our pod. (The
    Arditi case itself is handled by our wrapper's symlink — recognize
    the SHAPE: any hardcoded absolute path that won't exist on the pod.)

  - **USER-INPUT-NEEDED** — only the user can decide. Don't patch.
    Examples:
      * Two reasonable interpretations of intent.
      * Token / license / scope decisions.
      * Anything where the local Phase-1.5 checklist should have asked
        but didn't (the checklist needs to grow — note it in the
        blocker text so future iterations of the checklist absorb it).

Output ONE JSON object (no prose, no code fences). Schema:

{
  "verdict":   "ok" | "patches_applied" | "user_input_needed",
  "rationale": "<one short sentence>",
  "patches":   [
    {
      "file":    "<path relative to project root>",
      "find":    "<exact text to find — must appear exactly once in the file>",
      "replace": "<exact replacement text>",
      "reason":  "<short justification>"
    }
  ],
  "user_input_blockers": ["<one sentence per blocker>"]
}

- `verdict=ok` → empty `patches` and `user_input_blockers`.
- `verdict=patches_applied` → 1+ patches, empty blockers, all MECHANICAL.
- `verdict=user_input_needed` → 0+ patches (will be ignored), 1+ blockers.

If you have both mechanical patches AND user-input blockers, set
`verdict=user_input_needed` (we don't apply partial fixes when the user
also needs to weigh in). Resolution: report blockers, user resolves,
re-dispatch."""


PREP_USER_TEMPLATE = """Project under review:

- Repo URL: {project_repo_url}
- Branch:   {project_repo_branch}
- Pipeline: {pipeline_name}
- Target:   {target_model}
- Intent:   {intent}

Pipeline params: {params_json}

Project tree (depth=3, files only — first 100 entries):
{project_tree}

Key file excerpts (the pipeline class + the entrypoint script if findable):
{file_excerpts}

Apply the system-prompt's review rules. Return the JSON object."""


# --- file discovery + LLM-prompt prep --------------------------------------


def _list_project_tree(repo_root: Path, max_entries: int = 100) -> str:
    out: list[str] = []
    try:
        for p in sorted(repo_root.rglob("*"))[:max_entries]:
            if p.is_file():
                out.append(str(p.relative_to(repo_root)))
    except Exception:  # noqa: BLE001
        return "(failed to list project tree)"
    return "\n".join(out) if out else "(empty)"


def _read_excerpt(path: Path, max_chars: int = 4000) -> str:
    try:
        body = path.read_text(errors="ignore")
        return body[:max_chars] + ("\n...[truncated]" if len(body) > max_chars else "")
    except Exception:  # noqa: BLE001
        return f"(could not read {path})"


# --- LLM-output parsing + patch application -------------------------------


def _parse_plan(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction. Tolerates ``` fences and stray prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    # Last-resort: find the outermost {...} block in the text.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return None


def _apply_patch(repo_root: Path, patch: dict[str, Any]) -> tuple[bool, str]:
    """Apply one patch via verbatim find/replace. Returns (success, message)."""
    fname = patch.get("file", "")
    find = patch.get("find", "")
    replace = patch.get("replace", "")
    if not (fname and find):
        return False, "patch missing file or find"
    fpath = (repo_root / fname).resolve()
    # Guard against path traversal — keep edits inside the repo.
    try:
        fpath.relative_to(repo_root.resolve())
    except ValueError:
        return False, f"patch path {fname!r} is outside the project root; refusing"
    if not fpath.exists():
        return False, f"file not found: {fname}"
    body = fpath.read_text(errors="ignore")
    count = body.count(find)
    if count == 0:
        return False, f"find-text not present in {fname}"
    if count > 1:
        return False, f"find-text matches {count} sites in {fname}; refusing ambiguous patch"
    fpath.write_text(body.replace(find, replace, 1))
    return True, f"patched {fname} (1 site)"


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        return p.returncode, (p.stdout + p.stderr)[-1000:]
    except Exception as e:  # noqa: BLE001
        return 99, str(e)


def _commit_and_push(repo_root: Path, branch: str, run_id: str) -> tuple[bool, str]:
    """Best-effort: commit the applied patches and push the prepared branch.

    Uses PROJECT_REPO_TOKEN from env for HTTPS auth. Skips silently if there's
    no remote or no token configured.
    """
    if not (repo_root / ".git").exists():
        return False, f"no .git at {repo_root}"
    _run(["git", "config", "user.email", "autoresearch@noreply.local"], repo_root)
    _run(["git", "config", "user.name", "autoresearch"], repo_root)
    rc1, _ = _run(["git", "checkout", "-B", branch], repo_root)
    rc2, _ = _run(["git", "add", "-A"], repo_root)
    rc3, out3 = _run([
        "git", "commit", "-m",
        f"autoresearch: mechanical prep patches for run {run_id}",
    ], repo_root)
    token = os.environ.get("PROJECT_REPO_TOKEN")
    _, remote_url = _run(["git", "remote", "get-url", "origin"], repo_root)
    if token and remote_url.startswith("https://"):
        auth_url = remote_url.replace("https://", f"https://{token}@", 1).strip()
        _run(["git", "remote", "set-url", "origin", auth_url], repo_root)
    rc4, out4 = _run(["git", "push", "-u", "origin", branch], repo_root)
    ok = rc4 == 0
    return ok, f"checkout rc={rc1}; commit rc={rc3} {out3[-200:]}; push rc={rc4} {out4[-200:]}"


# --- entry point ----------------------------------------------------------


def prepare(
    run: Run,
    *,
    storage: StorageBackend,
    workspace: Path,
    model_client: ModelClient,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Single-shot review → apply mechanical patches → commit + push.

    Returns a result dict including `prepared_branch` (set when patches were
    applied + pushed) so transfer's prep gate can `git checkout` it.
    """
    project_root = project_root or (workspace / "project")
    pipeline_name = run.pipeline_name
    params = run.params or {}
    target = params.get("target_model", "—")
    intent = params.get("intent", "(none provided)")

    excerpts: list[str] = []
    pipeline_file = project_root / "pipelines" / f"{pipeline_name}.py"
    if pipeline_file.exists():
        excerpts.append(f"--- pipelines/{pipeline_name}.py ---\n{_read_excerpt(pipeline_file)}")
    reqs = project_root / "requirements.txt"
    if reqs.exists():
        excerpts.append(f"--- requirements.txt ---\n{_read_excerpt(reqs, max_chars=2000)}")

    user = PREP_USER_TEMPLATE.format(
        project_repo_url=params.get("project_repo_url", "—"),
        project_repo_branch=params.get("project_repo_branch", "—"),
        pipeline_name=pipeline_name,
        target_model=target,
        intent=intent,
        params_json=json.dumps(params, default=str),
        project_tree=_list_project_tree(project_root),
        file_excerpts="\n\n".join(excerpts) if excerpts else "(no excerpts available)",
    )

    agent = run_agent(
        run=run, storage=storage, client=model_client,
        system=PREP_SYSTEM, user=user, max_tokens=2000, label="prepare",
    )
    plan = _parse_plan(agent.text)

    # Output bookkeeping
    applied: list[str] = []
    failed: list[str] = []
    prepared_branch: str | None = None

    if plan is None:
        findings_mod.append(
            storage, run, FindingType.OBSERVATION,
            "[USER-INPUT-NEEDED] prep agent's output did not parse as JSON. "
            f"Raw output:\n{agent.text[:2000]}",
        )
        verdict = "user_input_needed"
    else:
        verdict = plan.get("verdict", "ok")
        blockers = plan.get("user_input_blockers") or []
        for b in blockers:
            # Loud user-input findings — leading "!" sorts them to the top.
            findings_mod.append(
                storage, run, FindingType.ERROR,
                f"![USER-INPUT-NEEDED] {b}",
            )

        if verdict == "patches_applied" and not blockers:
            patches = plan.get("patches") or []
            for p in patches:
                ok, msg = _apply_patch(project_root, p)
                tag = "applied" if ok else "FAILED"
                line = f"[MECHANICAL] {tag}: {p.get('file')} — {p.get('reason')}: {msg}"
                findings_mod.append(storage, run, FindingType.OBSERVATION, line)
                (applied if ok else failed).append(line)
            if failed:
                # Don't push partial sets. Tell the user.
                findings_mod.append(
                    storage, run, FindingType.ERROR,
                    "![USER-INPUT-NEEDED] some prep patches failed to apply; "
                    "not pushing branch. See preceding findings.",
                )
                verdict = "user_input_needed"
            elif applied:
                branch = f"autoresearch/prepared-{run.id}"
                ok, msg = _commit_and_push(project_root, branch, run.id)
                if ok:
                    prepared_branch = branch
                    findings_mod.append(
                        storage, run, FindingType.OBSERVATION,
                        f"[MECHANICAL] pushed `{branch}` ({len(applied)} patches). "
                        f"Transfer will check this out before running.",
                    )
                else:
                    findings_mod.append(
                        storage, run, FindingType.ERROR,
                        f"![USER-INPUT-NEEDED] patches applied locally but push failed: {msg}",
                    )
                    verdict = "user_input_needed"

    fresh = Run.load(storage, run.id)
    fresh.status = RunStatus.COMPLETED
    fresh.save(storage)
    return {
        "workflow": "prepare",
        "verdict": verdict,
        "patches_applied": applied,
        "patches_failed": failed,
        "prepared_branch": prepared_branch,
        "cost_usd": agent.cost_usd,
    }
