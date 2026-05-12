# Deploying autoresearch

End-to-end deploy: GitHub → image build (GHCR) → controller on Railway → pods on RunPod.

## One-time setup

### 1. Push the repo to GitHub

```bash
cd /Users/dmitrymanning-coe/Documents/Research/autoresearch
git init -b main
git add .
git commit -m "Initial v1"

# Create empty GitHub repo (web UI or gh CLI)
gh repo create autoresearch --private --source=. --remote=origin --push
```

The `.github/workflows/build-image.yml` will trigger automatically on push.

### 2. Wait for the first image build

GitHub Actions tab → "Build & push controller image". First build ~15 min (downloads the 10GB RunPod base). Pushes to `ghcr.io/<you>/autoresearch:sha-<short>` and `:latest`.

### 3. Make the package public (so RunPod can pull without auth)

GitHub → your profile → Packages → `autoresearch` → Package settings → Change visibility → Public.

(If you keep the package private, you'll need to give RunPod registry credentials — extra setup. Public is simpler for v1.)

### 4. Deploy the controller to Railway

```bash
brew install railway
railway login
cd /Users/dmitrymanning-coe/Documents/Research/autoresearch
railway init                       # creates a new Railway project
railway up                         # deploys from local; subsequent deploys via git push if you connect the repo
```

In Railway dashboard, set these env vars (under Variables):

```
MODE=serve
AWS_ACCESS_KEY_ID=<your R2 access key id>
AWS_SECRET_ACCESS_KEY=<your R2 secret>
AUTORESEARCH_RUNPOD_API_KEY=<your RunPod API key>
ANTHROPIC_API_KEY=<your anthropic key>
AUTORESEARCH_CONTROLLER_PUBLIC_URL=https://<your-project>.up.railway.app
AUTORESEARCH_STORAGE=s3
AUTORESEARCH_STORAGE_BUCKET=fra-proj
AUTORESEARCH_STORAGE_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
AUTORESEARCH_STORAGE_REGION=auto
AUTORESEARCH_COMPUTE=runpod
AUTORESEARCH_RUNPOD_NETWORK_VOLUME_ID=l7wyka84iy
AUTORESEARCH_RUNPOD_DEFAULT_IMAGE=ghcr.io/<you>/autoresearch:sha-<short>
AUTORESEARCH_DEFAULT_GPU=H100 80GB
# Optional:
# HF_TOKEN=<for gated models>
```

Verify the deploy:

```bash
curl https://<your-project>.up.railway.app/healthz       # {"status":"ok"}
curl https://<your-project>.up.railway.app/version       # {"version":"0.1.0"}
```

### 5. Wire local Claude Code to the controller

Add an entry to your MCP config (`~/.claude/config.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "autoresearch": {
      "type": "http",
      "url": "https://<your-project>.up.railway.app/mcp/mcp"
    }
  }
}
```

Restart Claude Code; the `start_transfer`, `list_runs`, etc. tools should appear.

## Per-project workflow

In any project that wants to use autoresearch:

```bash
uv add autoresearch
autoresearch init                   # drops .claude/skills/transfer.md, autoresearch.toml, pipelines/fra_example.py
```

Edit `autoresearch.toml` to point at your controller URL + R2 bucket + volume id, write your real Pipeline in `pipelines/`, and from Claude Code:

```
/transfer <pipeline-name> <target-model>
```

## Iterating on the package

Push a change → Actions builds a new image → Actions log prints the new `sha-<short>` tag → update `AUTORESEARCH_RUNPOD_DEFAULT_IMAGE` in Railway (controller picks it up for next pod spawn).

For controller-only changes (no pod-side impact), Railway can either rebuild from source or pull the new image — depending on how you connected it. Simplest: connect Railway to the GitHub repo so it auto-deploys on push.

## Verifying the full pipeline

Once the controller is live and Claude Code can see the MCP server:

1. From Claude Code: `/transfer fra_example Qwen/Qwen2.5-32B --budget 5`
2. RunPod dashboard: pod appears within ~30s, status RUNNING within 1-2 min.
3. Watch findings stream to your R2 bucket (`runs/<id>/findings/...`).
4. Cloudflare R2 dashboard: confirm `runs/<id>/run.json`, `checkpoint.json`, `heartbeat.json`, `findings/...`, `result.json` land as expected.
5. Test resilience: stop the pod externally (`runpodctl pod stop <id>`). Within ~5 min the supervisor logs (Railway log stream) should announce restarting; a new pod attaches the same volume; runner resumes from checkpoint.
6. `summarize_run(<run-id>)` via Claude Code returns a clean LLM digest.
