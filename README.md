# autoresearch

Automated-research toolkit for running ML experiments on cloud compute. Persistent execution, durable findings, MCP-accessible from Claude Code.

v1 ships one workflow real end-to-end:

- **TRANSFER** — run an existing user-defined pipeline against a different model (e.g. take the FRA measurement that works on Qwen-14B and run it on Qwen-32B).

Other workflows (replicate, sweep, multi-model review, night-run) exist as stubs and will be filled in iteratively.

## Architecture

- **Local Claude Code** — talks to the controller over MCP. Reads findings, tails logs, triggers takeover.
- **Controller** (FastAPI on Railway) — persistent process. Owns state (in S3), dispatches pods, watches heartbeats, restarts on death.
- **Compute** (RunPod) — long-lived GPU pods. The pipeline runner runs here.
- **Storage** (S3-compatible) — runs, findings, checkpoints, logs.

The "agent" role is small and structured: three bounded LLM calls (preflight check, postflight validation, error summarization). The pipeline runner itself is a deterministic FSM you can read in 60 lines.

## Quickstart (project side, once v1 is built)

```bash
uv add autoresearch
autoresearch init                       # drops .claude/skills/transfer.md and autoresearch.toml
# edit autoresearch.toml with your controller URL + S3 bucket
# write a Pipeline in pipelines/
```

From Claude Code on your laptop: `/transfer my_pipeline target_model`.

## Deploying

See `docs/deploy.md` for the full path: push to GitHub → image builds in CI → controller deploys to Railway → pods spawn on RunPod.

Short version:
1. Push the repo to GitHub. The `.github/workflows/build-image.yml` builds and pushes the image to `ghcr.io/<you>/autoresearch:sha-<short>`.
2. Make the GHCR package public (so RunPod can pull without auth).
3. Deploy the controller to Railway from this repo's Dockerfile (`MODE=serve`). Set R2 / RunPod / Anthropic env vars.
4. Add the controller to your local Claude Code's MCP config.
5. In any project, `uv add autoresearch && autoresearch init`, then `/transfer <pipeline> <model>`.

## Status

v1 code complete (96 tests passing). v1 verification (task 12) needs the GitHub→GHCR→Railway→RunPod loop wired live. See `notes/ideas.md` for the roadmap beyond v1.
