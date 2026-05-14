"""PREPARE workflow — pre-flight code review with REAL Claude Code on the pod.

Runs as a headless Claude Code session via `claude-agent-sdk`. The agent has
Read / Edit / Write / Bash / Glob / Grep tools and `cwd` set to the cloned
project repo on the persistent volume. Default outcome is still "no edits"
— most projects are fine. When the agent does find a fix, it applies it
directly via the Edit tool; we then `git status`, commit, and push to
`autoresearch/prepared-<run_id>` so the transfer pod's gate can check it out.

### What's different from the prior v0

  - **Real tools.** Multi-file edits, file creation, import smoke tests are
    all possible — they were not in the JSON-only version.
  - **No JSON contract.** The agent doesn't have to express patches as
    structured find/replace. It just edits.
  - **Audit trail is `git diff`.** The committed branch shows exactly what
    changed. Plus the SDK's tool-call stream becomes a finding so you can
    see what tools were used + with what inputs.

### What stays the same

  - Default outcome: "no edits, dispatch as-is."
  - User-input-needed escalation: agent prints `[USER-INPUT-NEEDED]` and
    halts (does NOT push); transfer's gate detects the tag and aborts.
  - Only edits inside the cloned project root (`cwd=/workspace/project`).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.core import findings as findings_mod
from autoresearch.core.agent_runner import run_agent_with_tools
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run, RunStatus


PREP_SYSTEM = """You are headless Claude Code doing pre-flight review of a
research project that's about to be dispatched onto a GPU pod. The
researcher is offline; whatever you decide is final until their next
check-in.

YOUR DEFAULT IS "no edits, ready to dispatch." Most projects are fine.

### Step 0 — disk hygiene on the network volume (DO THIS FIRST)

The pod's persistent volume is mounted at `/workspace`. Before anything
else, run `Bash: df -h /workspace` and check free space.

  - **≥50G free**: skip cleanup, move on to code review.
  - **<50G free**: prune before any other work, in this order:
      1. `du -sh /workspace/* 2>/dev/null | sort -h | tail -20` to find
         the biggest dirs.
      2. **HF cache** at `/workspace/.huggingface/hub/`. Each model lives
         at `models--<org>--<name>/`. Models pulled from HF Hub are
         durably stored upstream — local copies are pure cache. SAFE TO
         DELETE any model dir whose source repo on HF Hub is public and
         non-gated.
         Verify before deleting: `Bash: curl -sI https://huggingface.co/<org>/<name>`
         should return 200. If it 404s or 401s (gated / private),
         DO NOT delete — re-downloading may not be possible.
      3. **Old training outputs** at e.g. `/workspace/saes/`, `/workspace/outputs/`,
         `/workspace/runs/`. If you find a `*.uploaded` or
         `.hf_uploaded` marker, or the directory contains a `final/`
         step that matches a HF Hub repo name in any code you can grep
         (`Grep` for the dir basename across `pipelines/`), it's safe to
         delete. Otherwise leave it alone and note it as a finding —
         the user has to decide.
      4. **Pip cache** at `/workspace/.cache/pip/` is always safe to
         delete; it just slows the next install by 30-60s.
  - Print one line per deletion: `DELETED: /workspace/.huggingface/hub/models--X ( saved 12G )`.
  - End the cleanup phase with `Bash: df -h /workspace` again so the
    after-state is in the log.

If the floor in `disk_preflight.py` (currently 25G) trips AFTER your
cleanup, that's a hard failure — print `[USER-INPUT-NEEDED] disk floor
breached after cleanup; volume needs resize`.

### Step 1 — code review (the main job)

If you find a problem, categorize it:

  **MECHANICAL** — you can fix it yourself, right now, by editing files in
  the cwd. Use Read / Glob / Grep to find context, Edit / Write to make
  changes, Bash to verify (e.g. `python -c "import sae_lens"` after a
  requirements edit). Examples:
    - Hardcoded absolute paths in source (`/root/...`, `/home/...`,
      `/Users/...`, `/tmp/<user-specific>`) — patch to env-var or
      workspace-relative.
    - Missing or wrong dep pins (e.g. torch upgraded but torchvision left
      unpinned — we've hit this on Qwen-32B). Patch requirements.txt.
    - Concrete worked example from earlier work: Arditi's
      run_from_config.py hardcodes
      /root/git/dictionary_learning/data/misaligned_aggregated.jsonl —
      their checkout lives elsewhere on our pod. The wrapper-level fix is
      a symlink. Recognize the SHAPE: any hardcoded path with a username,
      hostname, or `/root/<repo>` prefix.

  **USER-INPUT-NEEDED** — only the researcher can decide. DO NOT EDIT.
  Print one line starting with `[USER-INPUT-NEEDED]` per blocker, then
  stop. Examples:
    - Ambiguous intent (two reasonable interpretations).
    - License / scope / cost decisions.
    - Anything the local Phase-1.5 checklist should have caught — note
      this in the blocker text so the checklist can grow.

  If you have BOTH mechanical fixes AND user-input blockers, set the
  user-input track only: print [USER-INPUT-NEEDED], do NOT edit. (Partial
  fixes confuse the audit trail. Resolution: researcher resolves blocker,
  re-dispatches, prep applies the mechanical fixes on the next pass.)

  Volume cleanup from Step 0 is independent — even if Step 1 surfaces a
  USER-INPUT-NEEDED blocker, the deletions you already made stay made.
  Note them in your reply so the audit trail is intact.

OUTPUT FORMAT:

  - If no edits needed: end with one line `OK: ready to dispatch` and one
    short sentence on what you confirmed. Include the freed-space delta
    if you cleaned anything.
  - If you applied mechanical edits: end with one line summary
    `EDITED: <N> files` and a one-line per-file changelog. Include the
    freed-space delta if you cleaned anything.
  - If user-input-needed: end with one or more
    `[USER-INPUT-NEEDED] <one-line description>` lines and nothing else.

Be terse. The researcher will read this on their next check-in."""


PREP_USER_TEMPLATE = """Project under review (cwd is the project root, with
the dispatch branch checked out):

  - Pipeline: {pipeline_name}
  - Target model: {target_model}
  - Researcher's intent: {intent}
  - Branch: {project_repo_branch}

Use Glob/Grep/Read to inspect. If you spot a MECHANICAL issue, fix it via
Edit/Write and verify if you can (Bash). If only the researcher can
decide, emit [USER-INPUT-NEEDED] and stop.

Defer to your system prompt for output format.
"""


def _run_git(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, check=False,
        )
        return p.returncode, (p.stdout + p.stderr)[-1000:]
    except Exception as e:  # noqa: BLE001
        return 99, str(e)


def _commit_and_push(repo_root: Path, branch: str, run_id: str) -> tuple[bool, str]:
    """git checkout -B + add -A + commit + push. Uses PROJECT_REPO_TOKEN."""
    if not (repo_root / ".git").exists():
        return False, f"no .git at {repo_root}"
    _run_git(["git", "config", "user.email", "autoresearch@noreply.local"], repo_root)
    _run_git(["git", "config", "user.name", "autoresearch"], repo_root)
    _run_git(["git", "checkout", "-B", branch], repo_root)
    _run_git(["git", "add", "-A"], repo_root)
    rc_c, out_c = _run_git([
        "git", "commit", "-m",
        f"autoresearch: prep agent edits for run {run_id}",
    ], repo_root)
    token = os.environ.get("PROJECT_REPO_TOKEN")
    _, remote_url = _run_git(["git", "remote", "get-url", "origin"], repo_root)
    if token and remote_url.startswith("https://"):
        auth_url = remote_url.replace("https://", f"https://{token}@", 1).strip()
        _run_git(["git", "remote", "set-url", "origin", auth_url], repo_root)
    rc_p, out_p = _run_git(["git", "push", "-u", "origin", branch], repo_root)
    return rc_p == 0, f"commit rc={rc_c} {out_c[-200:]}; push rc={rc_p} {out_p[-200:]}"


def prepare(
    run: Run,
    *,
    storage: StorageBackend,
    workspace: Path,
    model_client: ModelClient,        # unused; SDK uses env-side ANTHROPIC_API_KEY directly
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run the prep agent (headless CC). If it makes edits, push them."""
    project_root = project_root or (workspace / "project")
    params = run.params or {}

    user = PREP_USER_TEMPLATE.format(
        pipeline_name=run.pipeline_name,
        target_model=params.get("target_model", "—"),
        intent=params.get("intent", "(none provided)"),
        project_repo_branch=params.get("project_repo_branch", "—"),
    )

    agent = run_agent_with_tools(
        run=run, storage=storage,
        system=PREP_SYSTEM, user=user,
        cwd=project_root,
        allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
        max_turns=30,
        max_budget_usd=10.0,
        label="prepare",
    )

    # Categorize outcome from the agent's final-answer text.
    text = agent.text or ""
    user_input_needed = "[USER-INPUT-NEEDED]" in text
    prepared_branch: str | None = None

    if user_input_needed:
        findings_mod.append(
            storage, run, FindingType.ERROR,
            f"![USER-INPUT-NEEDED] prep agent halted; see prep finding above. "
            f"Transfer will abort at the gate.",
        )
        verdict = "user_input_needed"
    else:
        # Check working tree for edits the agent made. If any, push.
        rc, out = _run_git(["git", "status", "--porcelain"], project_root)
        has_changes = bool(out.strip()) and rc == 0
        if has_changes:
            branch = f"autoresearch/prepared-{run.id}"
            ok, msg = _commit_and_push(project_root, branch, run.id)
            if ok:
                prepared_branch = branch
                findings_mod.append(
                    storage, run, FindingType.OBSERVATION,
                    f"[MECHANICAL] pushed `{branch}` with prep edits. "
                    f"Transfer's gate will check this out before running.",
                )
                verdict = "patches_applied"
            else:
                findings_mod.append(
                    storage, run, FindingType.ERROR,
                    f"![USER-INPUT-NEEDED] prep edits made locally but git push failed: {msg}",
                )
                verdict = "user_input_needed"
        else:
            verdict = "ok"

    fresh = Run.load(storage, run.id)
    fresh.status = RunStatus.COMPLETED
    fresh.save(storage)
    return {
        "workflow": "prepare",
        "verdict": verdict,
        "prepared_branch": prepared_branch,
        "tool_calls": len(agent.tool_calls),
        "cost_usd": agent.cost_usd,
    }
