# soccer_hmm

Soccer players' external-load profile analysis with a Bayesian hidden Markov
model.

## Install

Python 3.12 or newer is required. From the repository root, activate the
Python environment you want to use, then install the package:

```bash
python -m pip install .
```

This installs the `soccer-hmm` command and the package's Python dependencies,
including CmdStanPy. You do not need to install CmdStanPy separately.

Or build distributable wheel and source archives with:

```bash
uv build --wheel
uv build --sdist
```

### Install CmdStan

CmdStanPy is the Python interface; fitting models also requires the CmdStan
program and a C++ build toolchain. Install them once, using the same Python
environment that provides `soccer-hmm`. On Windows, run:

```powershell
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(compiler=True)"
```

On Linux or macOS, install the required C++ compiler and `make` for your
platform, then run:

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Check which Python environment and CmdStan installation are being used with:

```bash
python -c "import sys, cmdstanpy; print(sys.executable); print(cmdstanpy.cmdstan_path())"
```

The CLI only attempts to install CmdStan when it cannot find an existing
installation. If CmdStan is already present but compilation reports that
`mingw32-make` cannot be found on Windows, the toolchain may be installed but
its executable directories are not on the current terminal's `PATH`. With the
default CmdStanPy installation location, add them to the current PowerShell
session with:

```powershell
$toolchain = Join-Path $env:USERPROFILE ".cmdstan\RTools40"
$env:PATH = "$toolchain\mingw64\bin;$toolchain\usr\bin;$env:PATH"
where.exe mingw32-make
where.exe g++
```

In Command Prompt, use:

```bat
set "RTOOLS=%USERPROFILE%\.cmdstan\RTools40"
set "PATH=%RTOOLS%\mingw64\bin;%RTOOLS%\usr\bin;%PATH%"
where mingw32-make
where g++
```

Both `where.exe` commands should print executable paths. Run `soccer-hmm` from
that same terminal session. If CmdStanPy installed RTools to a different
location, use that location instead of the default `RTools40` path above.

See the [CmdStanPy installation guide](https://mc-stan.org/cmdstanpy/installation.html)
for platform-specific requirements.

### Check the command

The package provides a command-line interface:

```bash
soccer-hmm --help
```

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
