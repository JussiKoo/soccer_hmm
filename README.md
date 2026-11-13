# soccer_hmm

Soccer players' external-load profile analysis with a Bayesian hidden Markov
model.

## Install

Python 3.12 or newer is required. Install the package and its Python
dependencies into an environment with:

```bash
python -m pip install .
```

Or build distributable wheel and source archives with:

```bash
uv build --wheel
uv build --sdist
```

The package provides a command-line interface:

```bash
soccer-hmm --help
```

CmdStan is required for fitting a model. `cmdstanpy` can install it on the
first run if it is not already available.

## Run an experiment

The package includes the default Stan model, but not player tracking data,
model outputs, or experiment configuration. Prepare a workspace with raw CSV
files and a model directory containing `config.yaml`, then pass their paths:

```bash
soccer-hmm \
  --raw-data-path "data/raw/your_data" \
  --processed-data-path "data/processed/soccerdata_processed.csv" \
  --model-path "models/your_model"
```

Use `--stan-path` to select a different Stan model.

The model configuration must define the state count, sampling settings, and
random seed. For example:

```yaml
model:
  name: soccer_hmm
  n_states: 5
sampling:
  chains: 4
  iter_warmup: 1000
  iter_sampling: 1000
  adapt_delta: 0.9
rng_seed: 42
```

The same paths are used by the `python -m soccer_hmm` command and the
`run_experiment.py` compatibility script. By default, project data, Stan
models, and experiment outputs are resolved relative to the current working
directory. Set `SOCCER_HMM_PROJECT_DIR` to use another project workspace.

## Python API

The `soccer_hmm` package includes data processing, modeling, diagnostics, and
plotting modules:

```python
from soccer_hmm.dataset import process_raw_data
from soccer_hmm.modeling.sample import fit_hmm
```
