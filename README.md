# soccer_hmm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Soccer players external load profile analysis with HMM's

Currently running an HMM experiment works as follows:

1. Create and activate the conda environment:
    conda create --file environment.yml
    conda activate hmm
3. Raw data files are in the "data/raw" directory as .CSV files (-raw_data_path)
4. Stan model is in the "stan" directory (-stan_path)
5. "models" directory contains a directory in which the config.yaml exists (-model_path)
   Example of config.yaml:
       model:
         name: soccer_hmm
         n_states: 5
       sampling:
         chains: 4
         iter_warmup: 1000
         iter_sampling: 1000
         adapt_delta: 0.9
       rng_seed: 42
6. Run the experiment from the project root:
   python -m run_experiment.py -raw_data_path "data/raw/your_data" -stan_path "stan/soccer_hmm_2.stan" -model_path "models/your_model"

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── models             <- MCMC sampling results for the experiments
├── notebooks          <- Jupyter notebooks
├── stan               <- Stan code files
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
        │        
        ├── model_checks.py      <- Contains functions for the model checks       
        └── sample.py            <- Contains functions to initialize and run the mcmc sampling
```

--------

