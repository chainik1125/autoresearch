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

ONLY flag a blocker when ONE of these is true (be precise about which):
  1. Hardcoded absolute paths that won't exist on the pod
     (`/root/...`, `/home/<user>/...`, `/Users/...`, project-specific tmp dirs).
  2. Environment variables the code reads that won't be set on the pod
     (anything beyond AWS_*, HF_TOKEN, ANTHROPIC_API_KEY, WANDB_API_KEY,
     PROJECT_REPO_TOKEN, RUN_ID, WORKSPACE_DIR, HF_HOME).
  3. Dependency manifest issues that pip will choke on (Mac-pinned
     wheels, missing transitive deps the project's imports need).
  4. Data files referenced but not in the repo or downloadable from HF Hub.

Output format:
- If no blockers: a single line "OK: looks fine, dispatch as-is" + one short
  sentence on what you confirmed.
- If blockers: one bullet per blocker, each with (a) file:line, (b) the line,
  (c) the proposed patch (text-level, e.g. "replace `/root/X` with
  `os.environ['WORKSPACE_DIR']/X`"). End with a one-line summary count.

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
