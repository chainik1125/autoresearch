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

## Quickstart (v0 plugin install)

`autoresearch` ships as a Claude Code plugin. The plugin gives you `/transfer`
+ `/make_compatible` skills and registers the MCP server pointing at your
controller — version-locked, no per-project copies, updates flow through
`/plugin update`.

In Claude Code:

```
/plugin marketplace add chainik1125/autoresearch
/plugin install autoresearch@autoresearch
```

When prompted, paste your controller URL (e.g.
`https://my-controller.up.railway.app/mcp/`).

Per-project setup is then just `autoresearch.toml`:

```bash
uv add autoresearch                     # for the Python package + CLI
# write autoresearch.toml manually or use `autoresearch init` for a stub
# write a Pipeline class in pipelines/<name>.py
```

From Claude Code on your laptop, in any project: `/transfer my_pipeline target_model`.

## Deploying

See `docs/deploy.md` for the full path: push to GitHub → image builds in CI → controller deploys to Railway → pods spawn on RunPod.

Short version:
1. Push the repo to GitHub. The `.github/workflows/build-image.yml` builds and pushes the image to `ghcr.io/<you>/autoresearch:sha-<short>`.
2. Make the GHCR package public (so RunPod can pull without auth).
3. Deploy the controller to Railway from this repo's Dockerfile (`MODE=serve`). Set R2 / RunPod / Anthropic env vars.
4. Add the controller to your local Claude Code's MCP config.
5. In any project, `uv add autoresearch && autoresearch init`, then `/transfer <pipeline> <model>`.

## Status

v1 verified end-to-end: Qwen-32B SAE training dispatched from Claude Code,
runs to completion, results back via MCP. 128 tests passing.

Plugin v0 (this commit): `/plugin marketplace add chainik1125/autoresearch`
ships `/transfer` + `/make_compatible` + the MCP server config in one
install. See `.claude-plugin/plugin.json` for the manifest;
`.claude-plugin/marketplace.json` for the marketplace stub.

See `notes/ideas.md` for the roadmap and `notes/working_notes.md` for the
current focus + the detailed failure-modes writeup from the Qwen-32B
canary loop (the nine root-cause bugs that drove v1 from "feature-
complete" to "actually-working").
