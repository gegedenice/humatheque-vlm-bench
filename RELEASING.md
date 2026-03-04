# Releasing ocr-bench

## Prerequisites

- Push access to `davanstrien/ocr-bench` on GitHub
- [Trusted publishing](https://docs.pypi.org/trusted-publishers/) configured on PyPI:
  - PyPI project: `ocr-bench`
  - Repository: `davanstrien/ocr-bench`
  - Workflow: `release.yml`
  - Environment: `pypi`
- GitHub environment `pypi` exists (Settings > Environments)
- Branch protection on `main` (require PR + passing CI)

## Release steps

1. **Create a release branch**:

   ```bash
   git checkout -b release/0.0.2
   ```

2. **Bump the version** in `pyproject.toml`:

   ```
   version = "0.0.2"
   ```

3. **Update the lockfile**:

   ```bash
   uv lock
   ```

4. **Run checks locally**:

   ```bash
   uv run ruff check src/ tests/
   uv run pytest tests/ -x -q
   ```

5. **Commit and push**:

   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Bump version to 0.0.2"
   git push -u origin release/0.0.2
   ```

6. **Open a PR** against `main`:

   ```bash
   gh pr create --title "Release 0.0.2" --body "Bump version to 0.0.2"
   ```

   CI runs automatically on the PR. Review and merge once green.

7. **Tag the release** (after merge):

   ```bash
   git checkout main && git pull
   git tag v0.0.2
   git push origin v0.0.2
   ```

   Or create a [GitHub Release](https://github.com/davanstrien/ocr-bench/releases/new) in the UI — set the tag to `v0.0.2` targeting `main`, which creates the tag and triggers the publish workflow.

8. **Watch the publish workflow**:

   ```bash
   gh run list --limit 1
   gh run watch <run-id>
   ```

9. **Verify**:

   ```bash
   pip install ocr-bench==0.0.2
   ocr-bench --help
   ```

## CI workflows

### `ci.yml` — runs on PRs and pushes to main

Lint (ruff) + tests (pytest). This is the check required by branch protection.

### `release.yml` — runs on `v*` tags

1. **test** job: same lint + tests as CI
2. **publish** job: build wheel > smoke test import > publish via trusted publishing

## Branch protection setup

Go to **Settings > Branches > Add rule** for `main`:

- Require a pull request before merging
- Require status checks to pass: select `test` (from the CI workflow)
- Do not allow bypassing the above settings (optional, but recommended)

## Troubleshooting

- **Lockfile out of sync**: CI uses `--locked`, so always run `uv lock` after changing `pyproject.toml`
- **Missing dev tool in CI**: Dev dependencies (ruff, pytest) must be in `[dependency-groups] dev`
- **test_web.py fails in CI**: Needs `--extra viewer` since FastAPI is optional
- **PyPI rejects upload**: PyPI doesn't allow re-uploading the same version. If publish failed after a partial upload, you'll need to bump to a new version.

## Versioning

[SemVer](https://semver.org/). Currently `0.0.x` (proof of concept). Bump to `0.1.0` when the API stabilises.
