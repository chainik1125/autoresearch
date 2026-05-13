"""PREPARE workflow — pre-flight static review of a research project.

Runs before the actual compute dispatch. Reads the cloned project repo, the
target Pipeline class, and the user's intent; reports any blockers for
unattended off-laptop dispatch.

### v0 behavior (single-shot, plan-mode-only)

  - One LLM call with the Read tool *conceptually* available via context
    (in v0 we paste relevant file snippets into the user prompt; v1 will
    enable real tool-using Read/Edit/Bash via the Claude Agent SDK).
  - Default action: **no changes.** The system prompt is explicit that the
    ideal outcome is "looks fine, dispatch as-is."
  - When the agent does flag a blocker, it WRITES the finding describing
    what it would patch, but does NOT actually patch the files. v1 will
    add the apply-patches + commit-to-branch path with explicit confirmation.

### v1 trajectory

  - Replace single-shot with Claude Agent SDK `query()` + Read/Edit/Bash
  - Add `autoresearch/prepared-<run_id>` branch creation + commit
  - Multi-turn confirmation: agent proposes patches, user (or a continuation
    agent) approves, agent applies + commits
  - Hook into make_compatible.md skill for the larger "wrap arbitrary
    project" case
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.core.agent_runner import run_agent
from autoresearch.core.run import Run, RunStatus


PREP_SYSTEM = """You are a research-engineering reviewer doing pre-flight static
analysis of a project that's about to be dispatched onto a GPU pod, off the
researcher's laptop. They're going to disengage as soon as the dispatch fires.

DEFAULT OUTCOME: "looks fine, dispatch as-is." Most projects do not need any
preparation — they're already in shape. Don't invent issues to look helpful.

Categorize each finding you produce as ONE of:

  - `[MECHANICAL]` — a fix the prep agent can/should apply itself without
    asking the user. The user has already disengaged. Examples:
      * Hardcoded absolute paths (`/root/...`, `/home/...`, `/Users/...`,
        project-specific tmp dirs) — propose a symlink or env-var-aware rewrite.
        Worked example we hit live: Arditi's `run_from_config.py` references
        `/root/git/dictionary_learning/data/misaligned_aggregated.jsonl`. Their
        clone lives at `/workspace/arditi_dl/...` on our pod. The fix is a
        symlink `/root/git/dictionary_learning -> /workspace/arditi_dl`. The
        wrapper in fra_proj's `fra/train_sae_arditi.py` does this in its
        `_ensure_hardcoded_symlink()`. Recognize this pattern: any hardcoded
        path with a username, hostname, or `/root/<repo>` shape.
      * Missing data files referenced by hardcoded paths — propose creating
        them from a downloadable HF Hub source if you can identify one.
      * Quirky module-level imports that need `sys.path` tweaks.
      * Dependency-manifest issues that pip's resolver would catch.
    For v0 you only REPORT [MECHANICAL] findings; v1 will apply + commit them.

  - `[USER-INPUT-NEEDED]` — a finding only the user can resolve. Examples:
      * Ambiguous design choices the local Claude's Phase 1.5 checklist
        should have caught but didn't (then the checklist needs to grow).
      * Two reasonable interpretations of what the user wants.
      * License / cost / scope decisions.
    Flag these LOUDLY — the user is disengaged; their next check-in is
    the only chance to resolve. Mark the finding's first character `!`
    so it sorts to the top of `list_findings`.

ONLY flag findings of either category when there's a real issue. Don't
invent things.

Output format:
- If no blockers: a single line "OK: looks fine, dispatch as-is" + one short
  sentence on what you confirmed.
- If blockers: one bullet per blocker, each with:
    (category) file:line — issue summary — proposed action
  End with a one-line summary count split by category.

Be terse. Do not write essays. Researchers will read your output."""


PREP_USER_TEMPLATE = """Project being prepared for dispatch:

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

Apply the system-prompt's review rules and produce the output."""


def _list_project_tree(repo_root: Path, max_entries: int = 100) -> str:
    """Plain `find`-style listing for the user prompt. Best-effort."""
    out: list[str] = []
    try:
        for p in sorted(repo_root.rglob("*"))[:max_entries]:
            if p.is_file():
                rel = p.relative_to(repo_root)
                out.append(str(rel))
    except Exception:  # noqa: BLE001
        return "(failed to list project tree)"
    return "\n".join(out) if out else "(empty)"


def _read_excerpt(path: Path, max_chars: int = 4000) -> str:
    try:
        body = path.read_text(errors="ignore")
        return body[:max_chars] + ("\n...[truncated]" if len(body) > max_chars else "")
    except Exception:  # noqa: BLE001
        return f"(could not read {path})"


def prepare(
    run: Run,
    *,
    storage: StorageBackend,
    workspace: Path,
    model_client: ModelClient,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run one pre-flight review pass. Writes findings, returns a result dict.

    `project_root` defaults to `workspace/project` (matches what the entrypoint
    clones to). Override for tests.
    """
    project_root = project_root or (workspace / "project")
    pipeline_name = run.pipeline_name
    target = (run.params or {}).get("target_model", "—")
    intent = (run.params or {}).get("intent", "(none provided)")

    # Best-effort: include the pipeline class file + a likely entrypoint snippet
    excerpts: list[str] = []
    pipeline_file = project_root / "pipelines" / f"{pipeline_name}.py"
    if pipeline_file.exists():
        excerpts.append(f"--- pipelines/{pipeline_name}.py ---\n{_read_excerpt(pipeline_file)}")

    user = PREP_USER_TEMPLATE.format(
        project_repo_url=(run.params or {}).get("project_repo_url", "—"),
        project_repo_branch=(run.params or {}).get("project_repo_branch", "—"),
        pipeline_name=pipeline_name,
        target_model=target,
        intent=intent,
        params_json=json.dumps(run.params, default=str),
        project_tree=_list_project_tree(project_root),
        file_excerpts="\n\n".join(excerpts) if excerpts else "(no excerpts available)",
    )

    agent = run_agent(
        run=run, storage=storage, client=model_client,
        system=PREP_SYSTEM, user=user, max_tokens=1500, label="prepare",
    )

    fresh = Run.load(storage, run.id)
    fresh.status = RunStatus.COMPLETED
    fresh.save(storage)

    looks_fine = agent.text.strip().lower().startswith("ok")
    return {
        "workflow": "prepare",
        "blockers_found": not looks_fine,
        "review_text": agent.text,
        "cost_usd": agent.cost_usd,
    }
