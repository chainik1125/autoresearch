"""POSTFLIGHT workflow — write experiment_summary.md + push, with real CC.

Fires after a transfer Run reaches a terminal state. Runs as a headless
Claude Code session via `claude-agent-sdk` with `cwd` set to the cloned
project repo. The agent has Read / Write / Bash tools; we hand it:

  - The target Run's params + status + result-as-json + findings dump
    (we write these into `cwd/.autoresearch_postflight_context.md` before
    invoking, so the agent can Read them).
  - A clear instruction: produce `autoresearch/runs/<target_run_id>/
    experiment_summary.md` with the spend in the headline, then commit
    + push to `autoresearch/results-<target_run_id>`.

After the agent returns, we verify the file exists. If not, we fall back
to the template-generated `experiment_summary.md` already produced by the
transfer workflow's terminal hook — so the user always gets *something*.

### Cleanup duties

The system prompt also tells the agent to:
  - Confirm HF Hub URLs in the result are accessible (Bash → `curl -sI`).
  - Note any artifact that's only on the volume (not durably stored).
  - Mark "compute released by supervisor" — the agent does NOT call
    cancel/terminate; the supervisor's cleanup pass reaps this pod
    after the workflow exits.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.config import Settings
from autoresearch.core import findings as findings_mod
from autoresearch.core import notify
from autoresearch.core.agent_runner import run_agent_with_tools
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run, RunStatus


POSTFLIGHT_SYSTEM = """You are headless Claude Code writing the post-run
summary for a research engineer who is OFF their laptop. Your output —
both the markdown report you write AND the git commit + push you make —
is what they'll see when they next check the project on their laptop or
GitHub.

You have Read / Write / Bash tools. cwd is the project repo (the
researcher's repo, with a checkout already in place; you'll create + push
to a new branch).

What to do, in order:

  1. Read `.autoresearch_postflight_context.md` (in cwd). It contains the
     target Run's metadata, params, result-as-JSON, and a chronological
     dump of its findings.

  2. Verify that durable artifacts named in the result are actually
     persisted to HF Hub / etc. — `Bash: curl -sI <hf_url>` etc. Note any
     that are missing.

  3. Write the experiment summary at `autoresearch/runs/<TARGET_RUN_ID>/experiment_summary.md`
     (replace TARGET_RUN_ID with the run id from the context file). The
     headline (REQUIRED) is:

         # /transfer <run_id> — <STATUS>
         **Total spend (est.): $X.XX** — LLM $Y.YY + compute ~$Z.ZZ. Budget $W.WW (under/EXCEEDED).
         **Outcome:** <one line — succeeded / failed / partial / aborted-by-prep-gate>.

     Then sections (terse, information-dense):
       ## What ran
       ## Result
       ## Artifacts — for each: where it lives + whether it's durably stored
       ## Loose ends — flagged things the user should follow up on
       ## Compute cleanup — note that the supervisor reaps this pod automatically

  4. Commit and push the new file to a branch
     `autoresearch/results-<TARGET_RUN_ID>`. Bash commands (use these
     exact lines so the audit trail is uniform):

         git config user.email "autoresearch@noreply.local"
         git config user.name "autoresearch"
         git checkout -B autoresearch/results-<TARGET_RUN_ID>
         git add autoresearch/runs/<TARGET_RUN_ID>/experiment_summary.md
         git commit -m "autoresearch: experiment summary for run <TARGET_RUN_ID>"
         git push -u origin autoresearch/results-<TARGET_RUN_ID>

     The wrapper has already configured PROJECT_REPO_TOKEN-backed HTTPS
     auth on `origin` for you. If push fails, write a finding noting
     what blocked you — don't retry forever.

  5. End your reply with one line: "DONE: pushed autoresearch/results-<TARGET_RUN_ID>"
     (or, on failure to push, "FAILED PUSH: <reason>")

Be terse. The summary doc should be readable in 30 seconds."""


POSTFLIGHT_USER_TEMPLATE = """Write the post-run summary for target_run_id={target_run_id}.

Context lives in .autoresearch_postflight_context.md (in cwd). Follow your
system prompt's procedure exactly.
"""


def _run_git(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        return p.returncode, (p.stdout + p.stderr)[-1000:]
    except Exception as e:  # noqa: BLE001
        return 99, str(e)


def _prep_repo_for_push(repo_root: Path) -> None:
    """Inject PROJECT_REPO_TOKEN into the `origin` URL so the agent's
    `git push` actually works."""
    token = os.environ.get("PROJECT_REPO_TOKEN")
    if not token:
        return
    _, remote_url = _run_git(["git", "remote", "get-url", "origin"], repo_root)
    if not remote_url.startswith("https://"):
        return
    auth_url = remote_url.replace("https://", f"https://{token}@", 1).strip()
    _run_git(["git", "remote", "set-url", "origin", auth_url], repo_root)


def _write_context_file(project_root: Path, target: Run, target_result: dict[str, Any] | None,
                        target_findings: list[Any]) -> Path:
    """Drop a markdown file with everything the agent needs to read."""
    lines = [
        f"# Target Run {target.id}",
        f"- workflow: {target.workflow}",
        f"- pipeline: {target.pipeline_name}",
        f"- status:   {target.status.value}",
        f"- params:   ```{json.dumps(target.params, default=str)}```",
        f"- budget:   spent=${target.budget_spent_usd:.4f} cap=${target.budget_cap_usd:.2f}",
        f"- last_error: {target.last_error}",
        "",
        "## Result",
        "```json",
        json.dumps(target_result, indent=2, default=str) if target_result else "(no result)",
        "```",
        "",
        f"## Findings ({len(target_findings)})",
    ]
    for f in target_findings[:50]:
        lines.append(f"\n### [{f.type.value} @ {f.created_at}]\n{f.body[:3000]}")
    ctx_path = project_root / ".autoresearch_postflight_context.md"
    ctx_path.write_text("\n".join(lines))
    return ctx_path


def postflight(
    run: Run,
    *,
    storage: StorageBackend,
    workspace: Path,
    model_client: ModelClient,        # unused; SDK reads ANTHROPIC_API_KEY from env
    target_run_id: str | None = None,
) -> dict[str, Any]:
    target_run_id = target_run_id or (run.params or {}).get("target_run_id")
    if not target_run_id:
        raise ValueError("postflight: missing target_run_id")

    project_root = workspace / "project"
    target = Run.load(storage, target_run_id)
    target_findings = findings_mod.list_findings(storage, target)
    try:
        target_result = json.loads(storage.read(target.result_key).decode("utf-8"))
    except Exception:  # noqa: BLE001
        target_result = None

    # Drop the context file the agent will Read, and arm origin for push.
    ctx_path = _write_context_file(project_root, target, target_result, target_findings)
    _prep_repo_for_push(project_root)

    user = POSTFLIGHT_USER_TEMPLATE.format(target_run_id=target_run_id)

    agent = run_agent_with_tools(
        run=run, storage=storage,
        system=POSTFLIGHT_SYSTEM, user=user,
        cwd=project_root,
        allowed_tools=["Read", "Write", "Bash"],
        max_turns=20,
        max_budget_usd=10.0,
        label="postflight",
    )

    # Verify the agent actually produced the file
    expected_md = project_root / "autoresearch" / "runs" / target_run_id / "experiment_summary.md"
    pushed = "pushed" in agent.text.lower() or expected_md.exists()
    branch = f"autoresearch/results-{target_run_id}"

    # Best-effort: ping the user's notification webhook if configured.
    # Loaded inline so we pick up env-var overrides on the pod (notification_url
    # comes from autoresearch.toml or AUTORESEARCH_NOTIFICATION_URL).
    notified = False
    try:
        settings = Settings.load()
        if settings.notification_url:
            headline = (
                f"{target.status.value.upper()} — spent "
                f"${target.budget_spent_usd:.2f} of ${target.budget_cap_usd:.2f}. "
                f"Results: autoresearch/results-{target_run_id}" if pushed
                else f"{target.status.value.upper()} — spent ${target.budget_spent_usd:.2f}. (summary push failed; see findings)"
            )
            notified = notify.send_notification(
                url=settings.notification_url,
                title=f"autoresearch /transfer {target_run_id}",
                message=headline,
                provider=settings.notification_provider,
            )
            findings_mod.append(
                storage, run, FindingType.OBSERVATION,
                f"notification → {settings.notification_provider or 'auto'}: "
                + ("sent" if notified else "FAILED (see logs)"),
            )
    except Exception:  # noqa: BLE001 -- notification must never break postflight
        pass

    fresh = Run.load(storage, run.id)
    fresh.status = RunStatus.COMPLETED
    fresh.save(storage)

    # Best-effort: clean up the context file from the working tree (it
    # shouldn't have been committed, but git status would show it).
    try:
        ctx_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass

    return {
        "workflow": "postflight",
        "target_run_id": target_run_id,
        "summary_pushed": pushed,
        "summary_branch": branch if pushed else None,
        "summary_path": str(expected_md) if expected_md.exists() else None,
        "notification_sent": notified,
        "tool_calls": len(agent.tool_calls),
        "cost_usd": agent.cost_usd,
    }
