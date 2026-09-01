# Robust DiM

Source code for Robust DiM.

Requires Python 3.13+. Use [uv](https://docs.astral.sh/uv/) for the virtual environment.

```bash
uv sync
uv run robustdim
```

## Files Architecture

```
src/: source code
tests/: test code
scripts/: sh files to run the code
logs/: log for different results
figs/: figs for experiments
data/: store different datasets and other data
configs/: yaml files for default configs
checkpoints/: store different checkpoints if has
README.md: this file
AGENTS.md: agent rules
pyproject.toml: project metadata and build config
.python-version: Python version pin
.gitignore: git ignore rules
.ignore: extra ignore rules
```
