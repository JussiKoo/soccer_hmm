import arviz as az
from cmdstanpy import from_csv
from src.config import MODELS_DIR

def load_fit(path):
    fit = from_csv(path=path)
    idata = az.from_cmdstanpy(posterior=fit)
    posterior = az.extract(idata, combined=False)
    return posterior