# Robust DiM

Source code for Robust DiM.

Requires Python 3.13+. Use [uv](https://docs.astral.sh/uv/) for the virtual environment.

```bash
uv sync
```

## Experiments

Do not run these until the config has been checked.

The judge runs `gemini-3.7-flash` on Vertex AI via Application Default Credentials (no API key). The 3.x models are served from the `global` endpoint only:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project nairr-260106-571547
```

```bash
export SAMBANOVA_API_KEY=...
uv run python -m robustdim.stability --dry-run
uv run python -m robustdim.tradeoff --dry-run
bash scripts/run_stability.sh
bash scripts/run_tradeoff.sh
```

Figures are written as PDF under `figs/`. JSON logs go under `logs/`.

## Files Architecture

```text
src/: source code
tests/: test code
notebooks/: jupyter notebooks
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
.ignore: ! un-ignore gitignored paths so agents can find them
LICENSE: MIT license text
```

## License

MIT. See [LICENSE](LICENSE).
