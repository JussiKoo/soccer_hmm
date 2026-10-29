# soccer_hmm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Soccer players external load profile analysis with HMM's

Currently running an HMM experiment works as follows:

1. Raw data files are in the raw-directory as .CSV files

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── models             <- MCMC sampling results for the experiments
├── notebooks          <- Jupyter notebooks
├── pyproject.toml     <- Project configuration file with package metadata for soccer_hmm and configuration for tools like black
├── environment.yml    <- The requirements file for reproducing the analysis environment
├── run_experiment.py  <- The script to run a single HMM experiment using the configurations in the models directory
└── src/soccer_hmm   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes soccer_hmm a Python module
    ├── config.py               <- Store useful variables and configuration
    ├── dataset.py              <- Contains functions for preprocessing the raw data
    ├── plots.py                <- Contains functions for plotting
    ├── modeling
    |   │        
    │   ├── model_checks.py      <- Contains functions for the model checks       
    │   └── sample.py            <- Contains functions to initialize and run the mcmc sampling
```

--------

