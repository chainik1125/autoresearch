"""autoresearch CLI — `autoresearch {run, serve, init}`.

`run`   — execute a workflow locally (no controller, no pod). The main entry
          point for development and for the on-pod runner.
`serve` — start the controller (task 9; stub in v1 so far).
`init`  — copy templates into the current project (task 10; stub in v1 so far).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from autoresearch.config import Settings, build_model_client, build_storage
from autoresearch.core.pipeline import Pipeline
from autoresearch.core.run import Run


def _load_pipeline(spec: str, pipeline_module_path: str) -> Pipeline:
    """Load a Pipeline class.

    `spec` is either:
      - "module.path:ClassName" — explicit import
      - "name" — search the configured pipeline_module_path for a module of that
        name, then find the class whose `name` attribute matches.
    """
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls()

    # Search the user's pipeline directory.
    search_root = Path(pipeline_module_path).resolve()
    if not search_root.exists():
        raise FileNotFoundError(
            f"pipeline_module_path {search_root} does not exist; either configure "
            f"`pipeline_module_path` in autoresearch.toml or pass an explicit "
            f"`module.path:ClassName` spec to --pipeline."
        )

    for path in search_root.rglob("*.py"):
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            continue
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and getattr(attr, "name", None) == spec:
                return attr()
    raise LookupError(
        f"No pipeline class with name={spec!r} found under {search_root}. "
        f"Use --pipeline module.path:ClassName for an explicit import."
    )


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.load()
    storage = build_storage(settings)
    model_client = build_model_client(settings)

    if args.run_id:
        run = Run.load(storage, args.run_id)
        # Pod-context terminal-state short-circuit.
        #
        # In pod mode (--heartbeat passed), if a prior attempt drove the run
        # to a terminal state, we exit 0 immediately so RunPod's
        # container-restart-on-non-zero-exit doesn't loop us forever on the
        # same failed pipeline. This defeated a 35-minute, $1.75 restart
        # cycle on the Qwen-32B canary.
        #
        # In local-CLI mode (no --heartbeat), we DON'T short-circuit — the
        # caller is explicitly asking us to resume, and the FSM handles
        # re-entering the failed phase from the checkpoint cleanly. This is
        # the path tests/test_cli.py:test_cli_run_resume_after_failure
        # exercises.
        if args.heartbeat:
            from autoresearch.core.run import RunStatus as _RS
            if run.status in (_RS.FAILED, _RS.COMPLETED):
                print(
                    f"run {run.id} already in terminal state {run.status.value}; "
                    f"exiting 0 so the pod can be cleaned up without retry.",
                    file=sys.stderr,
                )
                return 0
        workflow = run.workflow
        pipeline_name = run.pipeline_name
        print(
            f"resuming run {run.id} (workflow={workflow}, pipeline={pipeline_name})",
            file=sys.stderr,
        )
    else:
        if not (args.workflow and args.pipeline and args.target_model):
            print(
                "--workflow, --pipeline, and --target-model are required when --run-id is not given",
                file=sys.stderr,
            )
            return 2
        params: dict[str, Any] = {"target_model": args.target_model}
        if args.source_model:
            params["source_model"] = args.source_model
        if args.params:
            params.update(json.loads(args.params))
        workflow = args.workflow
        pipeline_name = args.pipeline
        run = Run(
            workflow=workflow,
            pipeline_name=pipeline_name,
            params=params,
            budget_cap_usd=args.budget or settings.default_budget_usd,
        )
        run.save(storage)
        print(f"created run {run.id}", file=sys.stderr)

    # Only transfer needs a Pipeline class. prep/mechanic/postflight are
    # agent-driven workflows that don't run user pipeline code.
    pipeline = _load_pipeline(pipeline_name, settings.pipeline_module_path) if workflow == "transfer" else None

    # Pod mode mounts the persistent volume at /workspace; local mode uses a per-run dir.
    default_workspace = os.environ.get("WORKSPACE_DIR") or f"./workspace-{run.id}"
    workspace = Path(args.workspace or default_workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # Workflow registry. Each workflow knows how to consume the inputs it
    # needs from the Run + storage + workspace.
    if workflow == "transfer":
        from autoresearch.workflows.transfer import transfer as run_workflow

        result = run_workflow(
            run, pipeline,
            storage=storage, workspace=workspace, model_client=model_client,
            preflight=settings.preflight and model_client is not None,
            postflight=settings.postflight and model_client is not None,
            summarize_errors=settings.summarize_errors and model_client is not None,
            heartbeat=args.heartbeat,
        )
    elif workflow in ("prepare", "mechanic", "postflight"):
        # Agent workflow gating differs by workflow:
        #   - mechanic uses run_agent_single_shot, which directly needs the
        #     ModelClient object (today: Anthropic only). Gate on it.
        #   - prepare + postflight use run_agent_with_tools, which delegates
        #     to a runner chain (ClaudeCodeRunner → CodexRunner). They only
        #     need at least ONE provider key in env, not the ModelClient
        #     object specifically. Gate on env vars directly.
        #
        # If the gate fails, exit 0 (NOT 2) so RunPod doesn't restart-loop
        # us on a structural config gap. Pin status=FAILED + ERROR finding
        # first so the user sees what happened.
        import os as _os
        from autoresearch.core import findings as findings_mod
        from autoresearch.core.findings import FindingType
        from autoresearch.core.run import RunStatus as _RS

        gate_failed_msg: str | None = None
        if workflow == "mechanic" and model_client is None:
            gate_failed_msg = (
                "workflow=mechanic requires ANTHROPIC_API_KEY to be set "
                "(mechanic is single-shot and uses the ModelClient directly). "
                "Add it on Railway, then redispatch."
            )
        elif workflow in ("prepare", "postflight"):
            has_anthropic = bool(_os.environ.get("ANTHROPIC_API_KEY"))
            has_openai = bool(_os.environ.get("OPENAI_API_KEY"))
            if not (has_anthropic or has_openai):
                gate_failed_msg = (
                    f"workflow={workflow} requires at least one of "
                    "ANTHROPIC_API_KEY (for claude-code runner) or "
                    "OPENAI_API_KEY (for codex fallback) to be set in pod "
                    "env. Add one on Railway, then redispatch."
                )

        if gate_failed_msg:
            print(gate_failed_msg, file=sys.stderr)
            findings_mod.append(storage, run, FindingType.ERROR, gate_failed_msg)
            fresh = Run.load(storage, run.id)
            fresh.status = _RS.FAILED
            fresh.last_error = "missing required API key(s) for this workflow"
            fresh.save(storage)
            return 0  # exit 0 -> RunPod doesn't restart-loop
        if workflow == "prepare":
            from autoresearch.workflows.prepare import prepare as run_workflow
        elif workflow == "mechanic":
            from autoresearch.workflows.mechanic import mechanic as run_workflow
        else:
            from autoresearch.workflows.postflight import postflight as run_workflow

        # Bracket the workflow call with diagnostic findings + a try/except.
        # Without this, any exception inside the workflow propagates out, the
        # process exits 1, RunPod restart-loops, and we have ZERO visibility
        # into what actually failed. Now we capture the traceback as an
        # ERROR finding, mark the Run FAILED, and exit 0 so RunPod stops
        # restarting. The user sees the failure via list_findings.
        from autoresearch.core import findings as findings_mod
        from autoresearch.core.findings import FindingType
        from autoresearch.core.run import RunStatus as _RS

        findings_mod.append(
            storage, run, FindingType.OBSERVATION,
            f"[cli] starting workflow={workflow} pipeline={pipeline_name}",
        )
        try:
            result = run_workflow(run, storage=storage, workspace=workspace, model_client=model_client)
        except Exception as exc:  # noqa: BLE001 -- capture-everything by design
            import traceback as _tb
            tb_text = _tb.format_exc()
            findings_mod.append(
                storage, run, FindingType.ERROR,
                f"workflow={workflow} crashed: {type(exc).__name__}: {exc}\n\n```\n{tb_text}\n```",
            )
            fresh = Run.load(storage, run.id)
            fresh.status = _RS.FAILED
            fresh.last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            fresh.save(storage)
            print(f"workflow {workflow} crashed: {exc}", file=sys.stderr)
            return 0  # exit 0 -> RunPod stops restart-looping
    else:
        # Structural config gap (unknown workflow). Same exit-0 treatment as
        # the missing-API-key path so we don't restart-loop on a typo.
        from autoresearch.core import findings as findings_mod
        from autoresearch.core.findings import FindingType
        from autoresearch.core.run import RunStatus as _RS
        msg = f"workflow {workflow!r} not implemented in v1"
        print(msg, file=sys.stderr)
        findings_mod.append(storage, run, FindingType.ERROR, msg)
        fresh = Run.load(storage, run.id)
        fresh.status = _RS.FAILED
        fresh.last_error = msg
        fresh.save(storage)
        return 0

    print(json.dumps(result, indent=2))
    return 0


def cmd_serve(_: argparse.Namespace) -> int:
    from autoresearch.controller.server import run as run_server

    run_server()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Copy templates into the current project.

    Idempotent: existing files are skipped unless --force is given.
    """
    import shutil
    from importlib.resources import files

    target_root = Path(args.target or ".").resolve()
    template_root = Path(__file__).resolve().parent.parent / "templates"

    if not template_root.exists():
        # Installed (non-editable) path: templates live alongside the package.
        # Fall back to importlib.resources for that case.
        try:
            template_root = Path(str(files("templates")))
        except (ModuleNotFoundError, FileNotFoundError):
            print(f"templates directory not found near {__file__}", file=sys.stderr)
            return 2

    pipelines_root = Path(__file__).resolve().parent.parent / "pipelines"
    plan = [
        (template_root / "autoresearch.toml.example", target_root / "autoresearch.toml"),
        (template_root / "skills" / "transfer.md", target_root / ".claude" / "skills" / "transfer.md"),
        (pipelines_root / "fra_example.py", target_root / "pipelines" / "fra_example.py"),
    ]

    copied = 0
    skipped = 0
    for src, dst in plan:
        if not src.exists():
            print(f"  ? source missing: {src}", file=sys.stderr)
            continue
        if dst.exists() and not args.force:
            print(f"  - skip (exists): {dst.relative_to(target_root)}", file=sys.stderr)
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  + {dst.relative_to(target_root)}", file=sys.stderr)
        copied += 1

    print(f"\ninit complete: {copied} copied, {skipped} skipped", file=sys.stderr)
    if copied > 0:
        print(
            "\nNext steps:\n"
            "  1. Edit autoresearch.toml: set controller_url, storage_bucket, runpod_network_volume_id.\n"
            "  2. Export secrets in your shell: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,\n"
            "     AUTORESEARCH_RUNPOD_API_KEY, ANTHROPIC_API_KEY (+ HF_TOKEN if you need gated models).\n"
            "  3. Replace pipelines/fra_example.py with your real pipeline.\n"
            "  4. From Claude Code: /transfer <pipeline-name> <target-model>",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute a workflow locally or on the pod")
    p_run.add_argument(
        "--workflow",
        default=None,
        help="workflow name (e.g. transfer); required unless --run-id is given",
    )
    p_run.add_argument(
        "--pipeline",
        default=None,
        help='pipeline name (searched in pipeline_module_path) or "module.path:ClassName"',
    )
    p_run.add_argument(
        "--target-model",
        default=None,
        help="model to run the pipeline against; required unless --run-id is given",
    )
    p_run.add_argument("--source-model", default=None, help="optional source model for comparison")
    p_run.add_argument("--budget", type=float, default=None, help="USD cap; default from config")
    p_run.add_argument(
        "--run-id",
        default=None,
        help="resume an existing run by id (loads workflow/pipeline/params from storage)",
    )
    p_run.add_argument(
        "--params",
        default=None,
        help="extra params as JSON dict; merged into the Run's params field",
    )
    p_run.add_argument(
        "--workspace",
        default=None,
        help="working directory; defaults to $WORKSPACE_DIR or ./workspace-<run_id>",
    )
    p_run.add_argument(
        "--heartbeat",
        action="store_true",
        help="spawn a heartbeat-writer thread (use for pod-side runs)",
    )
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve", help="start the controller (not yet implemented)")
    p_serve.set_defaults(func=cmd_serve)

    p_init = sub.add_parser("init", help="copy templates (skill + config + example pipeline) into a project")
    p_init.add_argument("--target", default=None, help="project directory; default = cwd")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.set_defaults(func=cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
