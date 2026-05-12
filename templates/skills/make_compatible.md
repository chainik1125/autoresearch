---
name: make_compatible
description: Adapt an existing research project to fit autoresearch's Pipeline protocol. Creates a branch in the user's project repo, wraps their existing measurement code as a Pipeline class, parameterizes the axis identified by Phase 0 of /transfer, drops an autoresearch.toml, and pushes the branch. Invoked by /transfer when the project's measurement code exists but isn't yet in the autoresearch-compatible shape.
---

# /make_compatible — wrap a project for the autoresearch pipeline

This skill is invoked from `/transfer` when Phase 0 scored the experiment
as `fit-with-adapter` — the user's measurement code exists but isn't yet a
`Pipeline` class autoresearch can dispatch.

Your job is to produce a branch in the user's project that:
1. Adds a `Pipeline` class at `pipelines/<name>.py` (the file the pod will load)
2. Adds or updates `autoresearch.toml`
3. Optionally adds `requirements.txt` if not present
4. Identifies the env vars the pipeline needs at runtime

You DO NOT modify `main`. You commit to a new branch named
`autoresearch/<workflow>-<short-slug>` (e.g. `autoresearch/transfer-qwen32b`).

You DO NOT push without explicit user confirmation.

---

## Inputs from `/transfer`

When `/transfer` invokes this skill, it has already established:

- **Project directory** — where the user's project lives on this laptop
- **Axis of variation** — `model` / `dataset` / `measurement` / `hyperparam` / `domain`
- **Pipeline name** — chosen by you or the user (snake_case slug)
- **Target value(s)** — what's being varied (e.g. `Qwen/Qwen2.5-32B`)
- **Baseline** — what stays fixed (e.g. the existing 14B result)

If any of these are missing, ask the user before proceeding. Don't guess on
the axis — it determines what you parameterize.

---

## Workflow

### 1. Restate the plan in one paragraph

Before touching anything, tell the user what you're about to do:

> "I'll create a branch `autoresearch/transfer-<slug>` off your current
> branch, add `pipelines/<name>.py` wrapping `<the existing entry point>`,
> parameterize `<axis>` so the pipeline accepts it via `params`, and create
> or update `autoresearch.toml`. I won't touch main, and I'll show you the
> diff before pushing. Proceed?"

Wait for confirmation. The user may correct the plan (different branch
name, different file location, different axis) — adjust before continuing.

### 2. Inspect the project

Read enough of the project to find:

**The entry point** — the script or function that currently runs the
measurement. Look at:
- `README.md` for usage examples
- Top-level `.py` files for `if __name__ == "__main__":`
- `pyproject.toml` for `[project.scripts]` entries
- `Makefile` or shell scripts under `scripts/`

If multiple entry points exist, ask the user which one. **Don't guess.**

**The hardcoded values on the chosen axis.** For axis=`model`, grep the
codebase for things like `"Qwen/Qwen2.5-14B"` or whatever name the user
mentioned. For axis=`dataset`, grep for dataset names / paths. For
axis=`measurement`, ask the user which call sites compute the measurement
to be replaced. Inspect every hit; some are documentation/comments and
don't need changing.

**The dependency manifest.** Check in order:
- `requirements.txt` at repo root — easiest, ready to use
- `uv.lock` or `poetry.lock` — extract a `requirements.txt` from it
- `pyproject.toml` `[project.dependencies]` — synthesize one
- If none of the above: inspect imports in the entry point + walk the
  immediate import tree, ask user to confirm

**Output structure.** Where does the existing script write results? Local
disk (`--output-root`)? WandB? S3? The Pipeline wrapper needs to redirect
small structured outputs to `storage.write()` and let heavy outputs land on
`workspace` (the persistent volume).

**Env vars used.** Grep for `os.environ.get`, `os.getenv`, and
`os.environ[`. Build a list of required env vars. Some may have
project-specific names (`MY_OPENAI_KEY` instead of `OPENAI_API_KEY`) —
note the mapping.

### 3. Create the branch

```
! cd <project-dir>
! git checkout -b autoresearch/<workflow>-<slug>
```

If a branch by that name already exists, append a short hash or date.
Don't reuse — that loses prior adaptation work and confuses git.

### 4. Draft the Pipeline class (show to user before writing)

The class lives at `<project-root>/pipelines/<name>.py`. Conform to the
protocol described in autoresearch's `notes/how-it-works.md`:

```python
class <ClassName>:
    name = "<name>"                          # matches the file name
    required_gpu = "<...>"                   # H100 80GB, A100 80GB, A40, etc.
    estimated_minutes = <int>                # rough; affects heartbeat threshold

    def run(self, *, params, workspace, storage):
        # Read the axis being varied from params:
        target_model = params["target_model"]    # for axis=model
        # source_model = params.get("source_model")  # optional

        # Call the existing measurement code with the new value substituted:
        result = <existing_function>(
            model_name = target_model,
            output_dir = workspace,
            # ... pass workspace for heavy state, storage for structured outputs ...
        )

        return result  # JSON-serializable dict
```

**Default to "wrap, don't restructure".** The Pipeline's `run()` calls into
existing code — it doesn't replace it. Two patterns:

- **(a) Import + call**: if the existing entry point exposes a callable
  function (`def main(args)` or similar), import and call it. Clean.
- **(b) Subprocess wrap**: if the existing entry point is a CLI-only script
  that mutates the world (writes files, parses argparse from `sys.argv`),
  shell it out with `subprocess.run([...])` and pass the new value via
  args. Less clean but doesn't require touching the original code.

(a) is preferred. Fall back to (b) only when (a) would require restructuring
the user's code (rare; most research scripts have *some* importable function).

**Show the proposed class to the user before writing it.** Include a brief
list of: which existing function/script it calls, which hardcoded value is
now read from `params`, and any compromises you made.

### 5. Write the file + update autoresearch.toml

Once the user confirms the draft:

- Write `pipelines/<name>.py`
- Create or update `autoresearch.toml`:
  - `pipeline_module_path = "pipelines"`
  - `project_repo_url = <https URL of the repo>`
  - `default_gpu = <required_gpu>` (or leave as the global default)
  - Any other fields the user already has — preserve them; only add what's
    missing.

If the project already has an `autoresearch.toml`, **don't overwrite**;
diff your changes and update only the relevant fields.

### 6. Handle dependencies

If `requirements.txt` doesn't exist next to `pipelines/`, generate one:

- From `uv.lock`: `uv export --format requirements-txt > requirements.txt`
- From `poetry.lock`: `poetry export -f requirements.txt > requirements.txt`
- From `pyproject.toml` dependencies block: synthesize manually

Tell the user what you generated and from what source. They can edit.

The pod entrypoint installs this once per network volume.

### 7. Document required env vars

Make a list. For each:

- The name the user's code expects (e.g. `OPENAI_API_KEY_MATS`)
- Whether it's already in `~/.zshenv` (check via `! printenv` or
  `! grep ^export ~/.zshenv | grep -i <pattern>`)
- Whether the controller already has it (you may not know without trying;
  err on the side of telling the user "I'll need this in Railway's env if
  it's not there already")

Tell the user the list at the end (Step 9 below).

### 8. Confirm + push

Show the user the full diff:

```
! cd <project-dir> && git status && echo "---" && git diff --stat HEAD
```

Walk through what changed. Ask explicitly:

> "Ready to commit and push this branch? I'll create a commit titled
> 'autoresearch: adapt <pipeline-name> for transfer' and push to
> `origin/autoresearch/<workflow>-<slug>`."

Wait for explicit "yes" before pushing. After:

```
! cd <project-dir>
! git add pipelines/ autoresearch.toml requirements.txt
! git commit -m "autoresearch: adapt <pipeline-name> for transfer"
! git push -u origin autoresearch/<workflow>-<slug>
```

### 9. Hand back to `/transfer`

Report a summary the user can scan in one breath:

> "Branch `autoresearch/<workflow>-<slug>` pushed.
>
> What `start_transfer` will dispatch:
>   pipeline_name = <name>
>   project_repo_branch = autoresearch/<workflow>-<slug>
>
> Env vars the pod will need:
>   ✓ AWS_ACCESS_KEY_ID (in your env, will pass through controller)
>   ✓ ANTHROPIC_API_KEY (in your env)
>   ⚠ OPENAI_API_KEY_MATS — your code expects this name. Make sure it's set
>      either in your env (and the controller will pick it up) or in the
>      controller's Railway env.
>
> Anything to adjust before I dispatch?"

If the user says go, the `/transfer` skill calls `start_transfer` with
the branch param.

---

## Edge cases to handle gracefully

### The project has no Pipeline-shaped function to wrap

The user's code is, say, a Jupyter notebook with all the logic in cells.
You can't import a notebook directly. Options:

- **(a)** Convert: use `jupyter nbconvert --to script` to produce a `.py`,
  edit to wrap the main logic in a function, then proceed.
- **(b)** Decline: tell the user "this project isn't structured in a way
  that wraps cleanly; you'll get a more reliable transfer if you first
  extract the measurement into a function." Fall back to a `mismatch`
  outcome.

(a) is invasive — only do it with user permission.

### The axis is "measurement" (different measurement on the same model)

This is harder than model-axis because the existing code may have the
measurement deeply tangled with everything else. Sub-cases:

- The new measurement is *additive* (compute it alongside the existing
  one): add it to the Pipeline's `run()`, return both numbers.
- The new measurement *replaces* the existing one: change the call site,
  parameterize via `params["measurement_name"]` if there are multiple
  swappable measurements.

Ask the user which sub-case before drafting.

### Project uses `uv` and the pod entrypoint uses `pip install`

For v1 our entrypoint script does `pip install --cache-dir ... -r requirements.txt`.
`uv` is faster but our entrypoint doesn't run `uv sync`. So we
*convert* the user's `uv.lock` to `requirements.txt` via
`uv export --format requirements-txt`. Note this in the report.

### Required env var has a project-specific name

If the user's code reads `OPENAI_API_KEY_MATS` and not `OPENAI_API_KEY`,
**don't** patch their code to look for the standard name — that's
restructuring, which we avoid. Instead, tell the user: "your code expects
`OPENAI_API_KEY_MATS`; make sure it's set in your shell env OR in the
controller's Railway env."

If the user prefers to standardize, that's a separate refactor they can do
later.

### The user wants to abort mid-flow

Treat any "stop" / "wait" / "let me think" as a signal to pause. Don't
commit. Don't push. Recap what's been done so far and what's pending; the
user can resume by re-invoking `/transfer`.

---

## What this skill explicitly does NOT do

- **Train models.** If a prerequisite (a new SAE, a new finetune, a new
  dataset) doesn't exist, that's research work outside autoresearch's scope.
  This skill bails to `/transfer`'s `mismatch` outcome.
- **Refactor user code beyond wrapping.** If the existing code is so tangled
  that wrapping requires substantial restructuring, decline with a
  `mismatch` outcome rather than producing a fragile wrapper.
- **Touch `main`.** Always a branch.
- **Push without confirmation.** Always ask.
- **Promise correctness of the wrapper.** The Pipeline class you generate is
  your best guess; it may have bugs. The first `/transfer` dispatch using it
  is also a test of the wrapper — the user should expect to iterate.
