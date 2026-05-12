# Working notes

Living document for in-progress autoresearch work. Distinct from
`how-it-works.md` (stable reference) and `ideas.md` (broader roadmap).

## Current focus (2026-05-12)

**Building the bridge layer for `/transfer`.** Going through the FRA use case
(Qwen-14B → Qwen-32B, from `chainik1125/fra_proj` on the `dmitry-em-repl`
branch) showed that the Pipeline protocol can't be a one-size-fits-all
interface for arbitrary user code — the gap should be bridged by a Claude
that adapts the user's existing project on the fly, not by forcing every
project to be manually restructured to fit the protocol.

**Approach (Path B):** Write a generic `templates/skills/make_compatible.md`
skill that instructs the local Claude how to adapt *any* project to
autoresearch's Pipeline protocol — create a branch, wrap the existing entry
point as a Pipeline class, parameterize hardcoded values, drop in
`autoresearch.toml`, check for missing env vars. Then test it on fra_proj as
the first real case and refine when we find gaps.

The `/transfer` skill is updated to detect-then-bridge-then-dispatch: ask the
controller `is_transfer_ready(project_repo_url, branch)`; if ready, dispatch;
if not, invoke `make_compatible.md`.

## TODOs

### Near-term (block FRA case)

- [ ] `templates/skills/make_compatible.md` — project-agnostic bridge skill
- [ ] `templates/skills/setup_check.md` — env-var verification
- [ ] MCP tool `is_transfer_ready(project_repo_url, branch)` on the controller
- [ ] Rewrite `templates/skills/transfer.md` to: detect → bridge → setup_check → dispatch
- [ ] Pod entrypoint: support `PROJECT_REPO_BRANCH` so dispatched pods can clone
      the bridge-created branch instead of the default branch
- [ ] **Private-repo cloning support.** Pod entrypoint accepts a
      `PROJECT_REPO_TOKEN` env (GitHub PAT) and rewrites the clone URL as
      `https://<token>@github.com/owner/repo.git`. Wire through
      `core/secrets.py` and `backends/compute/runpod.py`. *Required if
      fra_proj is private.*
- [ ] First end-to-end test: bridge Claude adapts `fra_proj` on a branch,
      `/transfer` dispatches against that branch, FRA runs on Qwen-32B.

### Long-term

- [ ] **Non-repo / tarball-upload path.** Users whose project has no git
      remote (or no push access) currently can't use `/transfer` at all.
      Support: package project as a tarball, upload to R2, pod fetches +
      extracts instead of `git clone`. ~Half a day of work; not urgent
      because every research repo we care about has a remote.
- [ ] Multi-pass `make_compatible`: bridge may need follow-ups (first attempt
      fails at `import`; retry with `sys.path` tweak). May want a small
      state machine or natural-language loop.
- [ ] Per-workflow bridge variants. `REPLICATE` and `SWEEP` will need
      different adaptation logic than `TRANSFER`. Either one skill per
      workflow or a meta-skill that branches at the top.
- [ ] `compare_runs(id_a, id_b)` MCP tool — diff result dicts + LLM-summarize
      the delta. ("Does Qwen-32B reproduce Qwen-14B's Δalign of +37.7?")
- [ ] Track non-Anthropic LLM spend. fra_proj's judge calls GPT-4o; that cost
      is invisible to autoresearch's budget primitive today.

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-12 | Path B (generic skill first, then test on fra_proj) over Path A | The artifact we actually want is the reusable skill; doing it concretely on fra_proj first risks producing fra_proj-shaped code rather than a general skill. |
| 2026-05-12 | Require git remote; defer tarball upload | Universal for research repos; tarball is real complexity for a v1 edge case. |
| 2026-05-12 | The bridge runs as the same local Claude that called `/transfer` | No new agent type. The local Claude already has Read/Edit/Bash on the project; just give it instructions. |
| 2026-05-12 | "Wrap, don't restructure" is the make_compatible default | Less invasive on user code. Fall back to refactor only if wrap is impossible. |

## Open questions

- **Is `chainik1125/fra_proj` public or private?** Determines whether
  private-repo cloning is a near-term blocker for the FRA case or just a
  forward-looking TODO.
- **Branch naming convention.** Lean `autoresearch/<workflow>-<target-slug>`
  (e.g. `autoresearch/transfer-qwen2.5-32b`) for grep-ability. Open to
  including a date or run-id prefix.
- **Push semantics for the bridge.** After it commits the wrapper to the new
  branch, should it `git push` automatically (after one explicit user
  confirmation), or always require the user to push themselves? Lean
  automatic-after-confirmation — fewer steps, still gated.

## Notes from the fra_proj exploration

For the bridge skill design, useful concrete observations from
`/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj` (dmitry-em-repl
branch):

- Entry point is an argparse script (`phase1_fra_orchestrator.py`) — not a
  class. Bridge needs to wrap, not just import.
- Model name is hardcoded in two places (`phase1_fra_orchestrator.py:37-61`,
  `phase1_additive_orchestrator.py:54-78`). Bridge needs to find all sites,
  not just one.
- Dependency manifest is `uv.lock`, not `requirements.txt`. Bridge or pod
  entrypoint needs to handle both formats (or generate `requirements.txt`
  from `uv export`).
- `OPENAI_API_KEY_MATS` is a project-specific env var name. Bridge should
  detect non-standard names and either re-export or note the mapping in
  `autoresearch.toml`.
- Memory: 32B model + LoRA merge peaks ~130GB during load. Even on H100 80GB
  this OOMs without `device_map="auto"` or sequential CPU-merge. The bridge
  may need to recommend (or auto-apply) this fix in the wrapped pipeline.
- The Phase 1 result requires an SAE trained on the target model (`Nura-J/Qwen2.5-14B_SAE_ln1.normalised` is 14B-specific). For 32B, either a 32B
  SAE must exist or the experiment shape changes. This is a *research*
  prerequisite, not a code one — autoresearch can't synthesize an SAE.
