"""MCP surface — the tools local Claude Code calls to drive runs.

`build_mcp(...)` constructs a FastMCP instance with tool closures that hold the
controller's storage / compute / model_client deps. The resulting instance is
mounted on the FastAPI app at `/mcp` (Streamable HTTP transport).

All tools handle `KeyNotFound` cleanly so calls to bogus run ids surface a clear
error instead of a 500.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from autoresearch.backends.compute import ComputeBackend
from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.config import Settings
from autoresearch.controller import dispatcher
from autoresearch.core import budget, findings, logs
from autoresearch.core.run import Run, RunStatus


_SUMMARY_SYSTEM = """You summarize automated research runs for a research engineer.
Be concise. Lead with one sentence: status + what was measured + key number(s).
Then 3-5 short bullets covering: notable observations, errors and likely cause
(if any), and how the result compares to expectations. Don't speculate beyond
what's in the findings."""


def build_mcp(
    *,
    settings: Settings,
    storage: StorageBackend,
    compute: ComputeBackend | None,
    model_client: ModelClient | None,
) -> FastMCP:
    mcp = FastMCP("autoresearch")

    @mcp.tool()
    def start_transfer(
        pipeline_name: str,
        target_model: str | None = None,
        source_model: str | None = None,
        gpu: str | list[str] | None = None,
        required_vram_gb: int | None = None,
        intent: str | None = None,
        budget_usd: float | None = None,
        params: dict[str, Any] | None = None,
        project_repo_url: str | None = None,
        project_repo_token: str | None = None,
        project_repo_branch: str | None = None,
    ) -> dict[str, Any]:
        """Start a TRANSFER run: dispatch a pod that runs `pipeline_name`.

        Params model (Phase 1):
          - `target_model` / `source_model` are *convenience shorthand* for
            the common model-swap case. They become entries in the run's
            params dict.
          - `params` is the *primary* way to specify what the pipeline
            receives. Anything Claude's intent-discovery determined needs
            to vary (hook_name, dataset, hyperparam, etc.) goes here.
          - The two are additive: explicit keys in `params` win over
            convenience args if both are passed for the same key.
          - At least one of `target_model` or `params` must be provided.

        Phase 2 (deferred, see working_notes.md): drop target_model/
        source_model from the signature, add baseline_params for general
        postflight comparison.

        `project_repo_token` is a GitHub PAT for private-repo cloning; overrides
        the controller's saved token for this dispatch only. The token lives in
        controller memory just long enough to inject into the pod env and is
        never persisted to storage or logs.

        `project_repo_branch` overrides the default branch so the pod clones a
        specific branch (e.g. one made by make_compatible.md).

        Hardware selection:
          - `gpu` can be a single GPU type name (e.g. "L40S") OR a list in
            preference order (e.g. ["L40S","L40","A6000","A40"]). Bypasses
            auto-selection.
          - `required_vram_gb`: when `gpu` is not given, the dispatcher
            queries the backend catalog and runs
            `core/hardware.py:recommend()` to pick. Combined with `intent`
            this uses the LLM advisor; without `intent` it falls back to
            the deterministic fastest-least-complicated heuristic.
          - `intent`: short free-form text describing what the run is for
            ("quick canary", "200M-token sweep, prefer cost"). Lets the
            advisor trade off speed vs cost. Prefer to call
            `recommend_hardware` first to surface confidence + ask the
            user if confidence is "needs_review"."""
        if compute is None:
            raise ValueError("compute backend not configured (set compute=runpod in autoresearch.toml)")

        final_params: dict[str, Any] = {}
        if target_model is not None:
            final_params["target_model"] = target_model
        if source_model is not None:
            final_params["source_model"] = source_model
        if params:
            # Explicit `params` keys override convenience args for the same key.
            final_params.update(params)

        if not final_params:
            raise ValueError(
                "must provide at least one of `target_model` or `params` "
                "(the pipeline needs something in its params dict)"
            )

        run = dispatcher.dispatch_new(
            workflow="transfer",
            pipeline_name=pipeline_name,
            params=final_params,
            budget_usd=budget_usd or settings.default_budget_usd,
            settings=settings,
            storage=storage,
            compute=compute,
            gpu=gpu,
            required_vram_gb=required_vram_gb,
            model_client=model_client,
            intent=intent,
            project_repo_url=project_repo_url,
            project_repo_token=project_repo_token,
            project_repo_branch=project_repo_branch,
        )
        return {"run_id": run.id, "status": run.status.value, "pod_handle": run.pod_handle}

    @mcp.tool()
    def list_runs(status: str | None = None) -> list[dict[str, Any]]:
        """List runs (newest first), optionally filtered by status."""
        runs = sorted(Run.list_all(storage), key=lambda r: r.created_at, reverse=True)
        if status:
            runs = [r for r in runs if r.status.value == status]
        return [
            {
                "id": r.id,
                "workflow": r.workflow,
                "pipeline": r.pipeline_name,
                "status": r.status.value,
                "created_at": r.created_at.isoformat(),
                "budget_spent_usd": r.budget_spent_usd,
                "budget_cap_usd": r.budget_cap_usd,
                "pod_handle": r.pod_handle,
            }
            for r in runs
        ]

    @mcp.tool()
    def get_run(run_id: str) -> dict[str, Any]:
        """Return the full Run record."""
        return Run.load(storage, run_id).model_dump(mode="json")

    @mcp.tool()
    def tail_log(run_id: str, lines: int = 200) -> str:
        """Tail the last N lines of the run's logs."""
        return logs.tail(storage, Run.load(storage, run_id), lines=lines)

    @mcp.tool()
    def list_findings(run_id: str, since_cursor: str | None = None) -> list[dict[str, Any]]:
        """List findings for a run, oldest first. Pass `since_cursor` (a finding
        storage key) to fetch only items strictly after it."""
        run = Run.load(storage, run_id)
        return [
            f.model_dump(mode="json")
            for f in findings.list_findings(storage, run, since_cursor=since_cursor)
        ]

    @mcp.tool()
    def summarize_run(run_id: str, max_tokens: int = 1200) -> str:
        """LLM-summarize a run's status + findings (server-side). Costs tokens —
        the spend is charged to the run's budget."""
        if model_client is None:
            raise ValueError("model_client not configured (set ANTHROPIC_API_KEY)")
        run = Run.load(storage, run_id)
        items = findings.list_findings(storage, run)
        compact = [
            {"type": f.type.value, "body": f.body[:1500]}
            for f in items
        ]
        user = (
            f"Run {run.id}\n"
            f"  workflow: {run.workflow}\n"
            f"  pipeline: {run.pipeline_name}\n"
            f"  status:   {run.status.value}\n"
            f"  params:   {json.dumps(run.params)}\n"
            f"  spent:    ${run.budget_spent_usd:.4f} of ${run.budget_cap_usd:.2f}\n\n"
            f"Findings ({len(items)}):\n{json.dumps(compact, indent=2)}"
        )
        resp = model_client.complete(system=_SUMMARY_SYSTEM, user=user, max_tokens=max_tokens)
        if resp.cost_usd > 0:
            budget.add_spend(storage, run, resp.cost_usd)
        return resp.text

    @mcp.tool()
    def get_budget(run_id: str) -> dict[str, float]:
        """Return current spend, cap, and remaining for a run."""
        run = Run.load(storage, run_id)
        return {
            "cap_usd": run.budget_cap_usd,
            "spent_usd": run.budget_spent_usd,
            "remaining_usd": budget.remaining(run),
        }

    @mcp.tool()
    def set_budget(run_id: str, new_cap_usd: float) -> dict[str, Any]:
        """Update the run's budget cap. Takes effect at the next FSM checkpoint."""
        run = Run.load(storage, run_id)
        run.budget_cap_usd = new_cap_usd
        run.save(storage)
        return {"ok": True, "cap_usd": run.budget_cap_usd}

    @mcp.tool()
    def takeover(run_id: str) -> dict[str, Any]:
        """Pause the run at the next checkpoint and return an SSH command for its pod.
        Use `release` to resume."""
        if compute is None:
            raise ValueError("no compute backend configured")
        run = Run.load(storage, run_id)
        if not run.pod_handle:
            raise ValueError(f"run {run_id} has no active pod")
        handle = compute.get_session(run.pod_handle)
        run.status = RunStatus.PAUSED
        run.save(storage)
        return {
            "ssh_command": handle.ssh_command(),
            "public_ip": handle.public_ip,
            "ssh_port": handle.ssh_port,
            "status": run.status.value,
            "note": "Effective at next FSM step boundary; in-flight tool call cannot be interrupted.",
        }

    @mcp.tool()
    def release(run_id: str) -> dict[str, Any]:
        """Resume a paused run. Supervisor's next heartbeat-stale tick will respawn
        the pod (if needed) and the runner resumes from checkpoint."""
        run = Run.load(storage, run_id)
        run.status = RunStatus.RUNNING
        run.save(storage)
        return {"ok": True, "status": run.status.value}

    @mcp.tool()
    def cancel(run_id: str) -> dict[str, Any]:
        """Terminate the pod, mark the run failed. Findings + checkpoint remain in storage."""
        run = Run.load(storage, run_id)
        if compute is not None and run.pod_handle:
            try:
                compute.terminate_session(run.pod_handle)
            except Exception:  # noqa: BLE001
                pass
        run.status = RunStatus.FAILED
        run.last_error = run.last_error or "cancelled by operator"
        run.save(storage)
        return {"ok": True, "status": run.status.value}

    @mcp.tool()
    def recommend_hardware(
        required_vram_gb: int,
        intent: str | None = None,
        pipeline_name: str = "<unspecified>",
        estimated_minutes: int = 0,
    ) -> dict[str, Any]:
        """Return a HardwareRecommendation for a hypothetical dispatch.

        The /transfer skill calls this in Phase 0 before `start_transfer` so
        it can:
          - Show the user the rationale upfront ("I picked H100 because…").
          - Branch on `confidence`: "high" → just dispatch and narrate;
            "needs_review" → ask the user via conversation, surfacing the
            `alternatives` so they have a choice.

        Returns: `{picks, confidence, rationale, alternatives}`. Picks is
        the gpuTypeId list (preference order) you'd pass to `start_transfer`
        as `gpu=...`. Confidence is "high" or "needs_review".

        See `core/hardware.py:recommend()` for the underlying logic.
        """
        if compute is None:
            return {
                "picks": [],
                "confidence": "needs_review",
                "rationale": "compute backend not configured; cannot query GPU catalog",
                "alternatives": [],
            }
        try:
            offers = compute.list_gpu_offers(settings.runpod_data_center)
        except Exception as exc:  # noqa: BLE001 -- surface error to caller
            return {
                "picks": [],
                "confidence": "needs_review",
                "rationale": f"failed to query GPU catalog: {exc!s}",
                "alternatives": [],
            }
        from autoresearch.core.hardware import recommend as _recommend
        rec = _recommend(
            offers,
            required_vram_gb=required_vram_gb,
            data_center_id=settings.runpod_data_center,
            intent=intent,
            pipeline_name=pipeline_name,
            estimated_minutes=estimated_minutes,
            client=model_client,
        )
        return {
            "picks": rec.picks,
            "confidence": rec.confidence,
            "rationale": rec.rationale,
            "alternatives": rec.alternatives,
        }

    @mcp.tool()
    def list_pipelines() -> list[str]:
        """List pipeline `name`s available under the configured pipeline_module_path."""
        root = Path(settings.pipeline_module_path)
        if not root.exists():
            return []
        names: set[str] = set()
        for path in root.rglob("*.py"):
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                continue
            try:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception:  # noqa: BLE001
                continue
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, type) and hasattr(obj, "name") and hasattr(obj, "run"):
                    name = getattr(obj, "name", None)
                    if isinstance(name, str):
                        names.add(name)
        return sorted(names)

    return mcp
