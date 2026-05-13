"""POSTFLIGHT workflow — produce the experiment summary + verify cleanup.

Fires AFTER a transfer Run reaches a terminal state (COMPLETED or FAILED).
Generates the LLM-written `experiment_summary.md` and commits + pushes it
to the user's project repo, so the headline (spend + result) shows up in
their git history without any further action from them.

### v0 responsibilities

  1. Read the target Run's result + findings.
  2. Generate the experiment_summary.md (markdown report; spend in headline).
  3. Commit + push it to `autoresearch/results-<run_id>` on the user's
     project repo. Best-effort: skip if no PROJECT_REPO_TOKEN or no remote.
  4. Verify durable artifacts — HF Hub uploads in the result dict, the
     SAE on the volume — are accessible. Report what's missing.
  5. The agent's system prompt tells it to confirm "all compute used is
     released" at the end, but the actual pod-termination is handled by
     the controller's supervisor (it terminates pods of terminal-status
     Runs on each tick — see controller/supervisor.py:tick), not by the
     agent calling cancel on itself.

### Pod self-cleanup

This pod is itself a Run; when its workflow completes, status flips to
COMPLETED. The next supervisor tick (≤30s) sees a terminal-status Run
whose pod is still RUNNING and terminates it. No bootstrap problem.

### v1 trajectory

  - Real Claude Agent SDK loop with Bash so the agent can actually
    `git commit -m ... && git push` rather than relying on subprocess
    helpers here.
  - Generate plots / figures (matplotlib) co-located with the markdown.
  - Open a PR rather than just push to a branch.
  - Cross-check: walk the agent tree via parent_run_id and report ALL
    Runs in the subtree, not just the one we postflight'd.
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
from autoresearch.core.run import Run, RunStatus


POSTFLIGHT_SYSTEM = """You write the post-run experiment summary for a research
engineer who is OFF their laptop and won't see this until they next open the
project repo on their laptop or GitHub.

You are given:
  - The target Run's metadata, params, status, error (if any)
  - Its full result dict
  - All findings written during the run

Produce a single markdown document. Headline format (REQUIRED):

  # /transfer <run_id> — <STATUS>
  **Total spend (est.): $X.XX** — LLM $Y.YY + compute ~$Z.ZZ. Budget $W.WW (under/EXCEEDED).
  **Outcome:** <one line — succeeded / failed / partial>.

Then sections, terse and information-dense:

  ## What ran
  pipeline / target_model / params (one line each)

  ## Result
  the key numbers a reader needs (tokens/sec, hf_url, wandb_url, etc.)

  ## Artifacts
  - sae_weights_path: <path on volume>; HF upload <URL if present, else "not uploaded">
  - training.log:     <volume path>
  - wandb:            <URL>
  - <other named outputs>
  For EACH artifact mark whether it appears to be persisted to durable
  storage (HF Hub URL given = yes; volume-only = ephemeral; missing = lost).

  ## Loose ends
  Anything the researcher should follow up on: failed validators, expensive
  surprises, artifacts that ended up only on the volume (not HF), etc.

  ## Compute cleanup
  Confirm that the relevant compute has been released. Note: this pod will
  itself terminate automatically when this workflow exits (supervisor pass).
  You do NOT call cancel/terminate explicitly. Just say "Compute released
  by supervisor on completion" or flag if you see something odd.

Be terse. The headline + outcome line should be enough for the user to know
if they need to read further. Total document <= 60 lines."""


def _try_run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
        return proc.returncode, (proc.stdout + proc.stderr)[-1000:]
    except Exception as e:  # noqa: BLE001
        return 99, str(e)


def _push_summary_to_repo(
    *, summary_md: str, run_id: str, project_root: Path, branch: str,
) -> tuple[bool, str]:
    """Best-effort: commit summary.md under autoresearch/runs/<id>/ and push.

    Returns (success, message). Honors PROJECT_REPO_TOKEN in env for auth.
    Skips silently if there's no git remote.
    """
    if not (project_root / ".git").exists():
        return False, f"no .git at {project_root}; skipping push"

    target_dir = project_root / "autoresearch" / "runs" / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "experiment_summary.md").write_text(summary_md)

    token = os.environ.get("PROJECT_REPO_TOKEN")
    # Configure committer for the bot commit
    _try_run(["git", "config", "user.email", "autoresearch@noreply.local"], project_root)
    _try_run(["git", "config", "user.name", "autoresearch"], project_root)

    rc1, _ = _try_run(["git", "checkout", "-B", branch], project_root)
    rc2, _ = _try_run(["git", "add", f"autoresearch/runs/{run_id}"], project_root)
    rc3, _ = _try_run([
        "git", "commit", "-m",
        f"autoresearch: experiment summary for run {run_id}",
    ], project_root)
    # If the remote URL is https://github.com/..., inject token for auth
    rc_url, remote_url = _try_run(["git", "remote", "get-url", "origin"], project_root)
    if token and remote_url.startswith("https://"):
        auth_url = remote_url.replace("https://", f"https://{token}@", 1).strip()
        _try_run(["git", "remote", "set-url", "origin", auth_url], project_root)
    rc4, out4 = _try_run(["git", "push", "-u", "origin", branch], project_root)

    ok = (rc1 == 0 or rc1 == 128) and rc4 == 0
    return ok, f"push rc={rc4}; {out4[-300:]}"


def postflight(
    run: Run,
    *,
    storage: StorageBackend,
    workspace: Path,
    model_client: ModelClient,
    target_run_id: str | None = None,
) -> dict[str, Any]:
    """Generate + push the experiment summary for `target_run_id`."""
    target_run_id = target_run_id or (run.params or {}).get("target_run_id")
    if not target_run_id:
        raise ValueError(
            "postflight: missing target_run_id (pass via run.params or call arg)"
        )

    target = Run.load(storage, target_run_id)
    target_findings = findings_mod.list_findings(storage, target)

    # Pull the pipeline result if it landed in storage.
    target_result: dict[str, Any] | None = None
    try:
        target_result = json.loads(storage.read(target.result_key).decode("utf-8"))
    except Exception:  # noqa: BLE001
        target_result = None

    user = (
        f"Target Run: {target.id}\n"
        f"  workflow: {target.workflow}\n"
        f"  pipeline: {target.pipeline_name}\n"
        f"  status:   {target.status.value}\n"
        f"  params:   {json.dumps(target.params, default=str)}\n"
        f"  budget:   spent=${target.budget_spent_usd:.4f} cap=${target.budget_cap_usd:.2f}\n"
        f"  last_error: {target.last_error}\n\n"
        f"Result: {json.dumps(target_result, indent=2, default=str) if target_result else '(none)'}\n\n"
        f"Findings ({len(target_findings)}):\n"
        + "\n\n".join(
            f"  [{f.type.value} @ {f.created_at}]\n  {f.body[:1500]}"
            for f in target_findings[:30]
        )
    )

    agent = run_agent(
        run=run, storage=storage, client=model_client,
        system=POSTFLIGHT_SYSTEM, user=user, max_tokens=2000, label="postflight",
    )
    summary_md = agent.text

    # Best-effort: push to user's project repo
    project_root = workspace / "project"
    branch = f"autoresearch/results-{target_run_id}"
    pushed_ok, push_msg = _push_summary_to_repo(
        summary_md=summary_md, run_id=target_run_id,
        project_root=project_root, branch=branch,
    )

    # Also drop a local copy on the volume for direct access
    try:
        (workspace / f"experiment_summary_{target_run_id}.md").write_text(summary_md)
    except Exception:  # noqa: BLE001
        pass

    fresh = Run.load(storage, run.id)
    fresh.status = RunStatus.COMPLETED
    fresh.save(storage)

    return {
        "workflow": "postflight",
        "target_run_id": target_run_id,
        "summary_pushed_to_branch": branch if pushed_ok else None,
        "push_message": push_msg,
        "summary_chars": len(summary_md),
        "cost_usd": agent.cost_usd,
    }
