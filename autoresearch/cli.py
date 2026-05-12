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

    pipeline = _load_pipeline(pipeline_name, settings.pipeline_module_path)

    # Pod mode mounts the persistent volume at /workspace; local mode uses a per-run dir.
    default_workspace = os.environ.get("WORKSPACE_DIR") or f"./workspace-{run.id}"
    workspace = Path(args.workspace or default_workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if workflow == "transfer":
        from autoresearch.workflows.transfer import transfer as run_workflow

        result = run_workflow(
            run,
            pipeline,
            storage=storage,
            workspace=workspace,
            model_client=model_client,
            preflight=settings.preflight and model_client is not None,
            postflight=settings.postflight and model_client is not None,
            summarize_errors=settings.summarize_errors and model_client is not None,
            heartbeat=args.heartbeat,
        )
    else:
        print(f"workflow {workflow!r} not implemented in v1", file=sys.stderr)
        return 2

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

    plan = [
        (template_root / "autoresearch.toml.example", target_root / "autoresearch.toml"),
        (template_root / "skills" / "transfer.md", target_root / ".claude" / "skills" / "transfer.md"),
        (template_root / "pipelines" / "fra_example.py", target_root / "pipelines" / "fra_example.py"),
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
