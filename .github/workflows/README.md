# GitHub Workflows

## Release Workflow

The release workflow automates the process of creating GitHub releases and publishing to PyPI.

### Usage

1. Go to Actions → Release → Run workflow
2. Fill in the inputs:
   - **Version**: The version number **without** 'v' prefix (e.g., `0.2.0`, `1.0.0-alpha`)
   - **Commit**: (Optional) Specific commit SHA to release. Defaults to latest main.
   - **Pre-release**: Check this for alpha/beta releases

### What it does

1. Creates a git tag (e.g., `v0.2.0`)
2. Builds the Python package using `uv` with dynamic versioning from the git tag
3. Creates a GitHub release with generated release notes
4. Publishes the package to PyPI

### Dynamic Versioning

This project uses `setuptools-scm` for dynamic versioning:

- Version is automatically determined from git tags
- No need to manually update version in `pyproject.toml`
- Development versions show as `0.2.0.dev5+gabcd123` between releases (how many commits since the last release + commit hash)
- The `verifiers/_version.py` file is auto-generated and not committed

### Requirements

- Repository secret `PYPI_API_TOKEN` must be set for PyPI publishing
- PyPi has strict semvar versions, no `v` prefix, solely `X.Y.Z` or `X.Y.Z-alpha`
- PyPi handles `-alpha` and `-beta` automatically. Example:
  - 0.12.0 is live and stable
  - publish 1.0.0-alpha to let people try 1.0.0

### Pre-releases

Pre-release versions (e.g., `1.0.0-alpha`) won't be installed by default:

- `uv add verifiers` → installs latest stable
- `uv add verifiers==1.0.0-alpha` → installs specific pre-release
- `uv add verifiers --prerelease=allow` → considers pre-releases
