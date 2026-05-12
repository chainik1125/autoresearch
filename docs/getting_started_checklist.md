# Getting started: the annoying-things checklist

Everything you need in place before `/transfer` just works, ordered by when you
hit it. Honest estimates on time, cost, and pain.

Long version with full UI walkthroughs is in `deploy.md`. This doc is the
"what am I in for?" overview.

## TL;DR

| Step | Time | Cost | Pain |
|---|---|---|---|
| 1. Sign up for 5 services | ~30 min | $0 upfront | low (forms) |
| 2. Generate credentials | ~30 min | $0 | **medium** — auth UIs are confusing |
| 3. Provision cloud resources | ~15 min | ~$14/mo ongoing | low (one-time) |
| 4. Deploy controller to Railway | ~20 min | ~$5/mo | low (first build is slow) |
| 5. Wire local Claude Code | 2 min | free | low |
| 6. Per-project setup (each new project) | ~10 min | free | low |
| **Total first-time** | **~2 hours** | **~$20/mo** | mostly waiting |

## Step 1 — Accounts (free signups)

All free to register. Some require a payment method even for free tier.

| Account | Why | Payment method required? |
|---|---|---|
| **GitHub** | Source code + image registry + project repo | no |
| **RunPod** | GPU pods | yes (~$10 minimum funding) |
| **Cloudflare** | R2 storage | yes (free tier still requires card on file) |
| **Railway** | Controller hosting | no (free $5/mo credit) |
| **Anthropic API** | Optional: validators (preflight / postflight LLM checks) + `summarize_run` | yes |
| **HuggingFace** | Probably already have one; only needs an account for gated models | no |

## Step 2 — Credentials (the painful one)

### 2a. GitHub Personal Access Token

**Settings → Developer Settings → Personal access tokens → Tokens (classic) → Generate new token.**

Scopes to check:
- ✅ `read:packages` — for RunPod to pull your container image from GHCR
- ✅ `repo` — for the pod to clone your private project repos AND for the bridge skill (`make_compatible.md`) to push adapted branches

Save the value somewhere; we'll register it with both RunPod and Railway. Typical convention: `export GIT_PAT="..."` in `~/.zshenv`.

**Sharp edges:**
- For organization-owned repos: after creating the token, click **Configure SSO** next to it and authorize it for each relevant org.
- Fine-grained PATs are more secure (scope to specific repos, granular permissions) but require more configuration. Default to classic; switch later if blast-radius bothers you.
- "Never expire" is convenient; some orgs require expiry. Up to you.

### 2b. RunPod API key

**RunPod dashboard → Settings → API Keys → Create API Key.**

One-click. Save it; typical convention: `export RP_API_KEY=...` in `~/.zshenv`.

### 2c. Cloudflare R2 token — **the worst one**

R2 has two distinct token types and the UI doesn't make this obvious. You need the **S3-compatible** kind, NOT the Cloudflare-API-Token kind. Walkthrough:

1. **Cloudflare dashboard → R2 Object Storage**. If first time, accept terms + add payment method.
2. **Create a bucket** (e.g. `my-project-state`).
3. **Note your Account ID** from the right sidebar; the S3 endpoint becomes
   `https://<account-id>.r2.cloudflarestorage.com`.
4. **R2 → Manage API Tokens → Create API Token**.
   - Permissions: **Object Read & Write**
   - Specify bucket: restrict to the one you just made
   - TTL: forever (or per your security policy)
5. The success screen shows **both** an Access Key ID and a Secret Access Key.
   **Copy both immediately** — the secret is shown only once.
6. Save:
   ```
   export AWS_ACCESS_KEY_ID="..."           # (yes, AWS_ — boto3 reads these names)
   export AWS_SECRET_ACCESS_KEY="..."
   ```

**Sharp edges:**
- The R2 UI also has a section that creates a Cloudflare API Token (Bearer-style, used for managing R2 via REST API). **Don't use that for S3 access** — boto3 can't authenticate with it.
- If you only see a single bearer token on the success page (no separate Access Key + Secret), you clicked the wrong "Create Token" flow. Go back to the R2-specific page.
- You can derive S3 credentials from a bearer-style R2 token via SHA-256 of the token value (Access Key ID = token id, Secret = sha256(value)) — but the clean S3-API token flow is simpler if you can find it.
- Bucket visibility: keep private (no public access needed).

### 2d. Anthropic API key (optional)

Only needed if you want validators (`preflight`, `postflight`, `summarize_run`).
Without it, the pipeline runs fine but you don't get the LLM safety/sanity
checks.

If you have an API key already (Claude Console → API Keys), `export ANTHROPIC_API_KEY=...` in `~/.zshenv`.

### 2e. HuggingFace token (optional)

Only needed for gated models (some Llama / Mistral variants). Public models
(Qwen, etc.) need nothing. If gated: HuggingFace → Settings → Access Tokens.
`export HF_TOKEN=...`.

## Step 3 — Cloud resources

### 3a. RunPod network volume

The persistent disk that holds your HF model cache + user pipeline state.
Without it, every pod cold-start has to redownload 60GB of model weights.

**Where**: RunPod dashboard → Storage → Network Volumes → Create.
- **Size**: 200GB recommended for Qwen-32B-class work (~$14/mo).
- **Data center**: pick one with the GPU type you'll use (e.g. `US-CA-2` has H100s).
- **Volume ID**: save it for `autoresearch.toml` (e.g. `vol_abc123`).

Cost: $0.07/GB/month, billed hourly while the volume exists (whether or not a
pod is attached).

### 3b. RunPod container-registry credential

Tells RunPod to use your GitHub PAT when pulling the container image (if your
image is private — which it is by default).

**Where**: RunPod dashboard → Settings → Container Registry Auth → Add new.
- Registry: `ghcr.io`
- Username: your GitHub username
- Password: your GitHub PAT from step 2a (the `read:packages` scope is what matters)

**Note the credential ID** (format: `cmp...`). Goes in `autoresearch.toml` as
`runpod_container_registry_auth_id`.

### 3c. GitHub Actions image build (push to trigger)

`.github/workflows/build-image.yml` is already in the repo. The first push to
`main` triggers a build (~15 min for the cold cache; ~2 min for incremental
builds afterward). The image lands at
`ghcr.io/<you>/autoresearch:latest`.

**Make the GHCR package public OR leave it private with the PAT registered (step 3b).**
Public avoids the registry-auth complication entirely; private is more secure.
Either works.

## Step 4 — Deploy controller to Railway

**Where**: Railway dashboard → New Project → Deploy from GitHub repo → pick `autoresearch`.

Set these env vars in Railway → Variables:

```
MODE=serve
AWS_ACCESS_KEY_ID=<from step 2c>
AWS_SECRET_ACCESS_KEY=<from step 2c>
AUTORESEARCH_STORAGE=s3
AUTORESEARCH_STORAGE_BUCKET=<your bucket>
AUTORESEARCH_STORAGE_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
AUTORESEARCH_STORAGE_REGION=auto
AUTORESEARCH_RUNPOD_API_KEY=<from step 2b>
AUTORESEARCH_RUNPOD_NETWORK_VOLUME_ID=<from step 3a>
AUTORESEARCH_RUNPOD_DEFAULT_IMAGE=ghcr.io/<you>/autoresearch:latest
AUTORESEARCH_RUNPOD_CONTAINER_REGISTRY_AUTH_ID=<from step 3b>
AUTORESEARCH_PROJECT_REPO_TOKEN=<from step 2a>
AUTORESEARCH_COMPUTE=runpod
AUTORESEARCH_DEFAULT_GPU=H100 80GB
AUTORESEARCH_CONTROLLER_PUBLIC_URL=https://<your-project>.up.railway.app
# Optional:
ANTHROPIC_API_KEY=<from step 2d>
HF_TOKEN=<from step 2e>
```

First deploy takes ~15 min (10GB ML base image). Verify with:
```bash
curl https://<your-project>.up.railway.app/healthz       # {"status":"ok"}
```

## Step 5 — Local Claude Code MCP wiring

```bash
claude mcp add --transport http --scope user autoresearch https://<your-project>.up.railway.app/mcp/
```

Restart Claude Code so the new server's tools register. Verify with `/mcp`
in Claude Code — `autoresearch` should appear with green/connected status.

## Step 6 — Per-project setup

Done once per research project you'll run pipelines for.

```bash
cd ~/my-project
uv add autoresearch
autoresearch init                                # drops .claude/skills/transfer.md, autoresearch.toml, pipelines/fra_example.py
```

Then edit `autoresearch.toml`:
- `controller_url` = your Railway URL
- `project_repo_url` = your GitHub repo URL (HTTPS, `https://github.com/you/your-project.git`)
- `storage_bucket` etc. — should match what you set on Railway

Your project must have a git remote (`git remote add origin ...`) and you must
have push access to it. The pod clones from there.

Write a real `Pipeline` class in `pipelines/` (replace the `fra_example` stub).
First `/transfer` from Claude Code.

## What's missing / wishlist

The above is hand-jammy. Three improvements we want eventually:

1. **`/setup-autoresearch` skill** — interactive walkthrough where Claude Code:
   - Checks which env vars are already set
   - Opens the right dashboard URLs at the right time
   - Detects "you already have a RunPod volume named X" instead of asking
   - Drives Cloudflare R2 token creation API-side (skipping the worst UI)
   - Sets Railway env vars via the Railway CLI (one command, not 12 dashboard pastes)
   - Registers the GHCR PAT with RunPod programmatically
   - Verifies the whole stack end-to-end before declaring done

2. **`/setup-doctor`** — a separate check skill that diagnoses an existing
   setup. Runs every check from step 6 and reports what's missing or broken
   (e.g. "your controller's running but `AUTORESEARCH_PROJECT_REPO_TOKEN` is
   not set, so private repos won't clone").

3. **Sane defaults**: today the project-side `autoresearch.toml` has 8+ values
   that need to match the controller's env. If the controller exposed a
   `get_settings_template()` MCP tool that returned the right toml for a given
   project, `autoresearch init` could fill in everything except the things
   that are genuinely user choices (pipeline name, GPU class). Reduces the
   per-project setup to "edit one line."

Until those exist, this checklist is the canonical path. The good news is
once you've done steps 1-5 once, you never touch them again — only step 6
recurs per project, and that's ~10 min.
