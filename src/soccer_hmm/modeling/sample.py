from pathlib import Path

from loguru import logger
#from tqdm import tqdm
import typer

#import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import cmdstanpy
from cmdstanpy import CmdStanModel
from cmdstanpy import from_csv
import os
from pathlib import Path
#import arviz as az
#import yaml
#from sklearn.cluster import KMeans

from src.soccer_hmm.config import PROCESSED_DATA_DIR, STAN_DIR, MODELS_DIR

def make_random_inits(template, n_states, rng):
    """Perturb the template inits randomly for one Pathfinder starting path."""
    inits = template.copy()

    inits["mu_speeds"] = [rng.uniform(0, 0.75), rng.uniform(0.75, 1.5), rng.uniform(1.5, 2.25), rng.uniform(2.25, 3), rng.uniform(3, 3.75)]
    inits["mu_loads"] = [rng.uniform(0, 2), rng.uniform(2, 4), rng.uniform(4, 6), rng.uniform(6, 8), rng.uniform(8, 10)]
    inits["sigma_speeds"] = rng.uniform(0.2, 1.0, size=n_states).tolist()
    inits["sigma_loads"] = rng.uniform(0.2, 1.0, size=n_states).tolist()
    inits["xangle"] = [rng.uniform(0.5, 2), rng.uniform(0.5, 2), rng.uniform(10, 12), rng.uniform(10, 12), rng.uniform(0.5, 2)]
    inits["yangle"] = rng.uniform(-1, 1, size=n_states).tolist()
    inits["beta"] = rng.uniform(-1, 1, size=(n_states * (n_states - 1), 2))

    return inits


def fit_hmm(model, processed_data, output_dir, model_configs):
    """
    Fits a HMM to the data
    
    Args: 
        model (CmdStanModel): Compiled model
        processed_data (pd.DataFrame): Processed data
        output_dir (Path): Output path for the sampling results
        model_configs (dict): Dictionary of the model configurations

    Returns:
        Fitted model as CmdStanMCMC object 
    """

    n_states = model_configs["model"]["n_states"]

    rng_seed = model_configs["rng_seed"]

    n_chains = model_configs["sampling"]["chains"]

    # Initializations for the chains
    inits_template = {
        "mu_speeds":    None,
        "sigma_speeds": None,
        "mu_loads":     None,
        "sigma_loads":  None,
        "xangle":       None,
        "yangle":       None,
        "beta":         None
    }
    

    #output_folder_name = "model-"+datetime.today().strftime('%d%m%Y-%H%M')

    X = np.column_stack([
        np.ones(processed_data.shape[0]),
        processed_data["accelrate_cumulative_30s_z"].to_numpy()
    ])

    modeling_data = dict({
        "T": processed_data.shape[0],
        "N_tracks": len(processed_data["track_id"].unique()),
        "track_ID": processed_data["track_id"].to_numpy(),
        "segment_ID": processed_data["segment_id"].to_numpy(),
        "speeds": processed_data["speed"].to_numpy(),
        "loads": processed_data["accelrate"].to_numpy(),
        "angles": processed_data["turn_angle"].to_numpy(),
        "N": n_states,
        "nCovs": 1,
        "covs": X
    })

    iter_warmup = model_configs["sampling"]["iter_warmup"]
    iter_sampling = model_configs["sampling"]["iter_sampling"]
    adapt_delta = model_configs["sampling"]["adapt_delta"]

    rng = np.random.default_rng(rng_seed)

    n_paths = 8
    path_inits = [make_random_inits(inits_template, n_states, rng) for _ in range(n_paths)]

    """
    n_mle = 10
    mle_inits = [make_random_inits(inits_template, n_states, rng) for _ in range(n_mle)]

    best_mle = None
    best_lp = -1e10

    for i in range(n_mle):
        try:
            mle = model.optimize(
                data=modeling_data,
                seed=rng_seed,
                inits=mle_inits[i],
                show_console=True,
                jacobian=True,
                require_converged=True,
                iter=2000
            )
        except RuntimeError as e:
            print(f"restart {i} failed to converge: {e}")
            continue

        lp = mle.optimized_params_dict["lp__"]
        if lp > best_lp:
            best_lp = lp
            best_mle = mle

    laplace_inits = model.laplace_sample(
        data=modeling_data,
        mode=best_mle,
        seed=rng_seed,
        draws=1000,
        jacobian=True,
        show_console=True
    ).create_inits(
        chains=n_chains,
        seed=rng_seed
    )
    """
    
    pathfinder_inits = model.pathfinder(
        data=modeling_data,
        num_paths=n_paths,
        num_single_draws=500,
        psis_resample=True,
        seed=rng_seed,
        inits=path_inits,
        num_threads=n_paths,
        show_console=True
    ).create_inits(chains=n_chains, seed=rng_seed)

    for i, init in enumerate(pathfinder_inits):
        print(f"Chain {i} inits:")
        print("-+-+-+-+-+-+-+-+-+-+")
        print(init)

    fit = model.sample(
        data=modeling_data,
        inits=pathfinder_inits,
        seed=rng_seed,
        chains=n_chains,
        parallel_chains=n_chains,
        threads_per_chain=1,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        output_dir=output_dir,
        show_progress=True
    )

    return fit

def load_model(model_folder):
    """
    Loads a CmdStanModel from a csv
    
    Args:
        model_folder (str): Folder name of the model
    
    Returns:
        CmdStanMCMC object

    Raises: 
        ValueError: If the model folder doesn't exist.
    """
    path = "./outputs/"+model_folder
    if (not os.path.exists(path)):
        raise ValueError("Not a valid model folder name.")
    fit = from_csv(path=path)
    return fit

app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    data_path: Path = PROCESSED_DATA_DIR / "soccerdata_processed.csv",
    stan_path: Path = STAN_DIR / "soccer_hmm.stan",
    exe_path: Path = STAN_DIR / "soccer_hmm.exe",
    models_path: Path = MODELS_DIR
    # -----------------------------------------
):

    return "Not implemented yet"
    # -----------------------------------------


if __name__ == "__main__":
    app()
