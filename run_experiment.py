from pathlib import Path

from loguru import logger
import typer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmdstanpy
from cmdstanpy import CmdStanModel
from cmdstanpy import from_csv
from pathlib import Path
import arviz as az
import scienceplots
from arviz_stats import summary
import yaml
import json
import shutil
from pprint import pformat

from src.soccer_hmm.config import PROCESSED_DATA_DIR, STAN_DIR, MODELS_DIR, RAW_DATA_DIR
from src.soccer_hmm.modeling.sample import fit_hmm
from src.soccer_hmm.dataset import process_raw_data
from src.soccer_hmm.modeling.model_checks import compute_all_pseudo_residuals, simulate_from_posterior_predictive, build_transition_matrix
from src.soccer_hmm.plots import (
    plot_autocorr, plot_emissions_subplot, plot_posteriors, plot_pp_density, plot_trace, 
    plot_transition_probs, plot_viterbi,
    psres_autocorr_plot, psres_histograms, psres_qqplot, plot_state_proportions
)

app = typer.Typer()

# Running from command line:
# python -m run_experiment --model-path "models/default" --no-load-chains

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    raw_data_path: Path = RAW_DATA_DIR,
    processed_data_path: Path = PROCESSED_DATA_DIR / "soccerdata_processed.csv",
    stan_path: Path = STAN_DIR / "soccer_hmm_2.stan",
    model_path: Path = MODELS_DIR / "default",
    load_chains: bool = False
    # -----------------------------------------
):

    plt.style.use(["science"])
    textwidth = 6.10356
    aspect_ratio = 1/3
    scale=2.0

    #Install cmdstanpy if not installed

    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        logger.info("Installing CmdStanPy")
        cmdstanpy.install_cmdstan()

    #==========================================================================================================

    #Instantiate a stan model from a stan-file.
    logger.info(f"Instantiating the model from {stan_path}")
    
    model = CmdStanModel(stan_file=stan_path, cpp_options={"STAN_THREADS":"true"})

    #Load configurations from a yaml-file
    logger.info(f"Loading configurations from {model_path}")

    config_filepath = model_path / "config.yaml"

    with open(config_filepath, "r") as f:
        model_configs = yaml.safe_load(f)

    #Copy stan-file and in the models directory
    if not load_chains:
        shutil.copy(stan_path, model_path / "soccer_hmm.stan")

    #Process raw data for modeling if not found
    if processed_data_path.exists():
        logger.info(f"Loading processed data from {processed_data_path}")

        processed_data = pd.read_csv(processed_data_path, sep=",", parse_dates=["timestamp"])
        scaling_params = {
            "accelrate_cumulative_30s": {
                "mean": processed_data["accelrate_cumulative_30s"].mean(),
                "std": processed_data["accelrate_cumulative_30s"].std()
        }
    }
    else:    
        logger.info(f"Processed data not found. Processing raw data in {raw_data_path}")
        processed_data, scaling_params = process_raw_data(raw_data_path, processed_data_path)

    #Save scaling parameters
    with open(model_path / "scaling_params.json", "w") as f:
            json.dump(scaling_params, f, indent=2)

    #==========================================================================================================

    #Modeling

    sampling_output_dir = model_path / "chains"

    #Load the model if load_chains==True
    if load_chains:
        logger.info(f"Loading posterior samples from {sampling_output_dir}")
        fit = from_csv(sampling_output_dir)

    else:
        logger.info("MCMC sampling starting soon...")

        fit = fit_hmm(
            model = model, 
            processed_data = processed_data, 
            output_dir = sampling_output_dir, 
            model_configs = model_configs
        )

        logger.info(f"Posterior samples saved to {sampling_output_dir}")

    n_states = model_configs["model"]["n_states"]
    rng_seed = model_configs["rng_seed"]
    rng = np.random.default_rng(rng_seed)

    #=====================================================================================================

    #Model diagnostics

    diagnostics_dir = model_path / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = fit.diagnose()

    with open(diagnostics_dir / "model_diagnostics.txt", "w") as f:
        f.write(diagnostics)

    logger.info(f"Model diagnostics saved to {diagnostics_dir / "model_diagnostics.txt"}")

    #====================================================================================================

    #Metadata

    config = dict(fit.metadata._cmdstan_config)

    # Don't include the sampled Stan variable names
    config.pop("column_names", None)
    config.pop("raw_header", None)

    metadata = pformat(config)

    with open(diagnostics_dir / "model_metadata.txt", "w") as f:
        f.write(metadata)



    #====================================================================================================

    #Summary statistics for posterior_draws

    idata = az.from_cmdstanpy(posterior=fit)
    idata.posterior = idata.posterior.drop_vars("viterbi")

    summary_stats = summary(idata, kind="all", ci_prob=0.95, ci_kind="eti", round_to=3)

    summary_file = diagnostics_dir / "params_summary.txt"
    summary_file_tex = diagnostics_dir / "params_summary_tex.txt"
    with open(summary_file, "w") as f:
        f.write(summary_stats.to_string())

    columns = ["mean", "sd", "eti95_lb", "eti95_ub", "ess_bulk", "ess_tail",  "r_hat"]
    header = ["Mean", "Std", "95\\% ci (lb)", "95\\% ci (ub)" , "Bulk-ESS" , "Tail-ESS" , "$\\hat{{R}}$"]
        
    with open(summary_file_tex, "w") as f:
        f.write(summary_stats.to_latex(
            columns=columns,
            header=header,
            float_format="%.3f"))

    logger.info(f"Parameter summary file saved to {diagnostics_dir}")

    #======================================================================================================

    #Marginal posterior density plots for the parameters

    posterior_draws = az.extract(idata, combined=True)

    plots_dir = model_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    vars_to_plot = ["shape_speeds", "rate_speeds", "shape_loads", "rate_loads", "loc", "kappa"]

    #marginal posterior density plots
    fig_posteriors = plot_posteriors(fit, 
                                     var_names=vars_to_plot,
                                     width=18,
                                     height_per_row=3)
    
    fig_posteriors.savefig(plots_dir / "posterior_densities.pdf")

    logger.info(f"Marginal posterior density plot saved to {plots_dir / "posterior_densities.pdf"}")

    #======================================================================================================

    #Plot traces of the chains

    fig_traces = plot_trace(fit, var_names=vars_to_plot)
    fig_traces.savefig(plots_dir / "traces.pdf")

    logger.info(f"MCMC trace plot saved to {plots_dir / "posterior_densities.pdf"}")

    #=====================================================================================================

    #design matrix and forecast pseudo-residuals

    dm = np.array([
        np.ones(processed_data.shape[0]), 
        processed_data["accelrate_cumulative_30s_z"].to_numpy()]
    )

    #Thinned draws for most model checks because of memory limitations
    shape_speeds = posterior_draws["shape_speeds"].values[:, ::10]
    rate_speeds = posterior_draws["rate_speeds"].values[:, ::10]
    shape_loads = posterior_draws["shape_loads"].values[:, ::10]
    rate_loads = posterior_draws["rate_loads"].values[:, ::10]
    loc = posterior_draws["loc"].values[:, ::10]
    kappa = posterior_draws["kappa"].values[:, ::10]
    beta = posterior_draws["beta"].values[:,:, ::10]

    logger.info("Computing the transition matrix...")

    tr_matrix = build_transition_matrix(
        betas=beta,
        design_matrix = dm,
        n_states=n_states
    )

    logger.info("Computing forecast pseudo-residuals...")

    speed_res, load_res, angle_res = compute_all_pseudo_residuals(
        processed_data["speed"].to_numpy(),
        processed_data["accelrate"].to_numpy(),
        processed_data["turn_angle"].to_numpy(),
        shape_speeds=shape_speeds,
        rate_speeds=rate_speeds,
        shape_loads=shape_loads,
        rate_loads=rate_loads,
        loc=loc,
        kappa=kappa,
        tr_matrix=tr_matrix,
        n_states=n_states
    )
    
    #plot forecast pseudo-residuals autocorrelations
    psres_autocorr_fig = psres_autocorr_plot(
        speed_res, 
        load_res, 
        angle_res, 
        track_id=processed_data["track_id"], 
        segment_id=processed_data["segment_id"],
        figsize=(scale*textwidth, scale*aspect_ratio*textwidth)
    )
    psres_autocorr_fig.savefig(plots_dir / "psres_autocorrelations.pdf")

    logger.info("Pseudo-residuals autocorrelation plot saved to plots directory.")

    #plot forecast pseudo-residuals qqplots

    qqplot_fig = psres_qqplot(
        speed_res, 
        load_res, 
        angle_res, 
        figsize=(scale*textwidth, scale*aspect_ratio*textwidth)
    )
    qqplot_fig.savefig(plots_dir / "pseudo_residual_qqplots.pdf")

    hist_fig = psres_histograms(speed_res, load_res, angle_res, bins=30, figsize=(21,7), agg="mean")
    hist_fig.savefig(plots_dir / "hist_fig.pdf")

    logger.info("Pseudo-residuals qqplot saved to plots directory.")
    

    #configs for emissions distribution plots
    configs = [
        {
            "scipy_dist": "gamma",
            "posterior": posterior_draws,
            "dist_params": ("shape_speeds", "rate_speeds"),
            "xlabel": "Meters per second (m/s)",
            "xlim": (0, processed_data["speed"].max()),
            "ylim": (0, 2),
            "title": "Speed"
        },
        {
            "scipy_dist": "gamma",
            "posterior": posterior_draws,
            "dist_params": ("shape_loads", "rate_loads"),
            "xlabel": "Arbitrary units (AU)",
            "xlim": (0, processed_data["accelrate"].max()),
            "ylim": (0, 1.4),
            "title": "Accel'Rate"
        },
        {
            "scipy_dist": "von_mises",
            "posterior": posterior_draws,
            "dist_params": ("kappa", "loc"),
            "xlabel": "Degrees ($^\\circ$)",
            "xlim": (-np.pi, np.pi),
            "ylim": (0, 1.4),
            "title": "Turn angle"
        }
    ]

    #plot emission distributions
    emissions_fig = plot_emissions_subplot(
        configs, 
        n_states=n_states, 
        plot_ci=True, 
        figsize=(scale*textwidth, scale*aspect_ratio*textwidth)
    )
    emissions_fig.savefig(plots_dir / "emission_distributions.pdf")

    logger.info("Emission distributions plot saved to plots directory.")

    #==================================================================================================================

    logger.info("Simulating from posterior predictive.")

    ppspeeds, pploads, ppangles, ppstates = simulate_from_posterior_predictive(
        T = processed_data.shape[0], 
        segment_id=processed_data["segment_id"].values,
        track_id=processed_data["track_id"].values,
        shape_speeds=shape_speeds,
        rate_speeds=rate_speeds, 
        shape_loads=shape_loads, 
        rate_loads=rate_loads, 
        loc_angles=loc,
        kappa_angles=kappa,
        beta=beta,
        global_mean=scaling_params["accelrate_cumulative_30s"]["mean"],
        global_sd=scaling_params["accelrate_cumulative_30s"]["std"],
        rng=rng,
        window_samples=15
    )
    
    
    posterior_pred_sim_fig = plot_pp_density(
        speed=processed_data["speed"],
        accelrate=processed_data["accelrate"],
        turn_angle=processed_data["turn_angle"],
        pp_samples_speed=ppspeeds,
        pp_samples_accelrate=pploads,
        pp_samples_turn_angle=ppangles,
        figsize=(scale*textwidth, scale*aspect_ratio*textwidth)
    )

    posterior_pred_sim_fig.savefig(plots_dir / "posterior_predictive_sims.pdf")

    logger.info("Posterior predictive simulation plot saved to plots directory.")

    #=====================================================================================================================

    #autocorrelation plots for observed data and posterior predictive samples

    autocorr_fig = plot_autocorr(
        processed_data["speed"].values,
        processed_data["accelrate"].values,
        processed_data["turn_angle"].values,
        processed_data["track_id"].values,
        processed_data["segment_id"].values,
        pp_samples_speed=ppspeeds,
        pp_samples_accelrate=pploads,
        pp_samples_turn_angle=ppangles,
        lags=30,
        figsize=(scale*textwidth, scale*aspect_ratio*textwidth)
    )
    
    autocorr_fig.savefig(plots_dir / "autocorrelations.pdf")

    logger.info("Autocorrelation plot saved to plots directory.")

#=====================================================================================================================

    #logger.info("Simulating from prior predictive.")

    """
    prior_sims = simulate_from_priors(
        rng=rng,
        n_draws=n_draws,
        n_states=n_states
    )

    prior_pred_speeds, prior_pred_loads, prior_pred_angles, prior_pred_states = simulate_from_posterior_predictive_2(
        T = processed_data.shape[0], 
        segment_id=processed_data["segment_id"].values,
        track_id=processed_data["track_id"].values,
        shape_speeds = prior_sims["shape_speeds"],
        rate_speeds = prior_sims["rate_speeds"], 
        shape_loads = prior_sims["shape_loads"], 
        rate_loads = prior_sims["rate_loads"], 
        loc_angles = prior_sims["loc"],
        kappa_angles = prior_sims["kappa"],
        beta=prior_sims["beta"],
        global_mean=scaling_params["accelrate_cumulative_30s"]["mean"],
        global_sd=scaling_params["accelrate_cumulative_30s"]["std"],
        rng = rng,
        window_samples=15
    )    


    prior_pred_sim_fig = plot_pp_density(
        speed=processed_data["speed"],
        accelrate=processed_data["accelrate"],
        turn_angle=processed_data["turn_angle"],
        pp_samples_speed=prior_pred_speeds,
        pp_samples_accelrate=prior_pred_loads,
        pp_samples_turn_angle=prior_pred_angles
    )
    """

    """

    prior_model = CmdStanModel(stan_file= STAN_DIR / "soccer_hmm_priors_only.stan")

    prior_fit = prior_model.sample(
        data={"T": processed_data.shape[0], "N": n_states, "n_covs": 1},
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,   # 1000 total prior draws is plenty for a predictive check
        seed=rng_seed,
    )

    pp_sim_model = CmdStanModel(stan_file=STAN_DIR / "hmm_pp_sim.stan")

    prior_pp = pp_sim_model.generate_quantities(
        data={
            "T": processed_data.shape[0], "track_ID": processed_data["track_id"], 
            "segment_ID": processed_data["segment_id"],
            "N": n_states, "n_covs": 1,
            "global_mean": scaling_params["accelrate_cumulative_30s"]["mean"], 
            "global_sd": scaling_params["accelrate_cumulative_30s"]["std"],
            "window_samples": 15,
        },
        previous_fit=prior_fit,   # reuses the prior draws, no resampling needed
    )

    prior_pred_speeds = prior_pp.stan_variable("pp_speeds")   # shape (n_draws, T)
    prior_pred_loads = prior_pp.stan_variable("pp_loads")
    prior_pred_angles = prior_pp.stan_variable("pp_angles")
    prior_pred_states = prior_pp.stan_variable("pp_states")

    prior_pred_sim_fig = plot_pp_density(
        speed=processed_data["speed"],
        accelrate=processed_data["accelrate"],
        turn_angle=processed_data["turn_angle"],
        pp_samples_speed=prior_pred_speeds,
        pp_samples_accelrate=prior_pred_loads,
        pp_samples_turn_angle=prior_pred_angles
    )

    prior_pred_sim_fig.savefig(plots_dir / "prior_predictive_fig.pdf")
    """

    #====================================================================================================

    #transition probability plots

    n_bins = 100
    cov_sequence_z = np.linspace(
        start=np.min(processed_data["accelrate_cumulative_30s_z"]), 
        stop=np.max(processed_data["accelrate_cumulative_30s_z"]), 
        num=n_bins
    )

    mean = scaling_params["accelrate_cumulative_30s"]["mean"]
    std = scaling_params["accelrate_cumulative_30s"]["std"]
    
    cov_sequence = cov_sequence_z*std + mean

    dm = np.array([np.ones(n_bins), 
                   cov_sequence_z])

    #Use all posterior draws here
    beta = posterior_draws["beta"].values

    logger.info("Computing the transition matrix...")

    tr_matrix = build_transition_matrix(
        betas=beta,
        design_matrix=dm,
        n_states=n_states
    )

    transition_probs_fig = plot_transition_probs(
        cov=cov_sequence,
        tr_matrix=tr_matrix,
        figsize=(scale*textwidth, 2.0*scale*aspect_ratio*textwidth),
        plot_ci=True
    )
    
    transition_probs_fig.savefig(plots_dir / "transition_probs.pdf")

    logger.info(f"Transition probabilities plot saved to {plots_dir / "transition_probs.pdf"}")

    #=====================================================================================================

    #viterbi path
    df_vit = fit.draws_pd(vars=["viterbi"], inc_warmup=False).astype(int)
    vit = df_vit.mode(axis=0).iloc[0] #Mode for each time step
    vit_np = vit.T.to_numpy()

    #Cumulative states visualizations
    state_proportions_fig = plot_state_proportions(track_id=processed_data["track_id"],
                                                  segment_id=processed_data["segment_id"],
                                                  player_id=processed_data["player_id"],
                                                  game=processed_data["game"],
                                                  viterbi=vit_np,
                                                  timestamp=processed_data["timestamp"],
                                                  n_states=n_states,
                                                  figsize=(scale*textwidth, 2*scale*aspect_ratio*textwidth))
    state_proportions_fig.savefig(plots_dir / "state_proportions.pdf")

    logger.info("State proportions plot saved to plots directory.")

    #=====================================================================================================

    #Viterbi path visualizations

    viterbi_fig = plot_viterbi(
        obs=processed_data, 
        viterbi=vit_np, 
        n_states=n_states,
        figsize=(scale*textwidth, scale*textwidth)
    )
    
    viterbi_fig.savefig(plots_dir / "viterbi_paths.pdf")

    logger.info("Viterbi paths plot saved to plots directory.")

    #====================================================================================================



if __name__ == "__main__":
    app()
