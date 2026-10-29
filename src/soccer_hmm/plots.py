from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import scipy.stats as stats
import matplotlib.colors as mcolors
from scipy.stats import gamma, vonmises, norm
import math
import seaborn as sns
import pandas as pd
import matplotlib
#from statsmodels.graphics.tsaplots import acf
from scipy.stats import gaussian_kde
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from statsmodels.tsa.stattools import acf
from scipy.stats import circmean

from src.soccer_hmm.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def plot_posteriors(fit, var_names=None, width=18, height_per_row=3, **kwargs):
    """Plot posterior densities and return the Matplotlib figure.

    Args:
        fit: CmdStanPy fit object to convert to ArviZ inference data.
        var_names: Posterior variable names to plot. If ``None``, plot all
            variables in the posterior.
        width: Figure width in inches.
        height_per_row: Height of each subplot row in inches.
        **kwargs: Additional keyword arguments passed to
            :func:`arviz.plot_posterior`.

    Returns:
        matplotlib.figure.Figure: Figure containing the posterior plots.
    """
    idata = az.from_cmdstanpy(fit)

    if var_names is None:
        var_names = list(idata.posterior.data_vars)

    # Count scalar posteriors
    n_plots = 0
    for var in var_names:
        da = idata.posterior[var]
        n_plots += da.size // (da.shape[0] * da.shape[1])

    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    axes = az.plot_posterior(
        idata,
        var_names=var_names,
        grid=(n_rows, n_cols),
        figsize=(width, height_per_row * n_rows),
        round_to=2,
        **kwargs,
    )

    # Retrieve the figure from the first Axes
    fig = axes.flat[0].figure

    return fig


def plot_trace(fit, var_names):
    """Plot MCMC trace diagnostics for the selected variables.

    Args:
        fit: Inference data or sampling result accepted by
            :func:`arviz.plot_trace`.
        var_names: Names of the variables to include in the trace plots.

    Returns:
        matplotlib.figure.Figure: Figure containing the trace plots.
    """
    axes = az.plot_trace(fit, var_names=var_names, figsize=(20,40))
    fig = axes.flat[0].figure

    return fig


def psres_qqplot(speed_fpres, load_fpres, angle_fpres, figsize=(21,7)):
    """Plot QQ envelopes for forecast pseudo-residuals.

    Args:
        speed_fpres: Array of speed pseudo-residuals, with observations on the
            first axis and draws on the second.
        load_fpres: Array of load pseudo-residuals, with observations on the
            first axis and draws on the second.
        angle_fpres: Array of angle pseudo-residuals, with observations on the
            first axis and draws on the second.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing the three QQ plots.
    """
    fig, axs = plt.subplots(1,3, figsize=figsize, sharey=True)

    #stats.probplot(speed_fpres.mean(axis=1), plot=axs[0])
    #stats.probplot(load_fpres.mean(axis=1), plot=axs[1])
    #stats.probplot(angle_fpres.mean(axis=1), plot=axs[2])

    qq_envelope(speed_fpres, axs[0])
    qq_envelope(load_fpres, axs[1])
    qq_envelope(angle_fpres, axs[2])

    for ax in axs:
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
    
    axs[0].set_title("Speed")
    axs[1].set_title("Accelrate")
    axs[2].set_title("Turn angle")

    fig.tight_layout()
    return fig


def psres_histograms(speed_fpres, load_fpres, angle_fpres, bins=30, figsize=(21,7), agg="mean"):
    """Plot pseudo-residual histograms with standard normal density overlays.

    Args:
        speed_fpres: Speed pseudo-residuals, as a 1D array, a ``(T, n_draws)``
            array, or ``None``.
        load_fpres: Load pseudo-residuals, as a 1D array, a ``(T, n_draws)``
            array, or ``None``.
        angle_fpres: Angle pseudo-residuals, as a 1D array, a ``(T, n_draws)``
            array, or ``None``.
        bins: Number or specification of histogram bins accepted by
            :meth:`matplotlib.axes.Axes.hist`.
        figsize: Figure size as ``(width, height)`` in inches.
        agg: For 2D inputs, ``"mean"`` averages draws per observation;
            ``"flatten"`` pools all draws. Other values use the mean behavior.

    Returns:
        matplotlib.figure.Figure: Figure containing the three histograms.
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, residuals, title in zip(
        axs,
        [speed_fpres, load_fpres, angle_fpres],
        ["Speed", "Accel'rate", "Turn angle"]
    ):
        # build 1D array of residuals to plot
        if residuals is None:
            vals = np.array([])
        else:
            arr = np.asarray(residuals)
            if arr.ndim == 2:
                if agg == "flatten":
                    vals = arr.ravel()
                else:
                    vals = arr.mean(axis=1)
            else:
                vals = arr.ravel()

        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        # histogram (density)
        ax.hist(vals, bins=bins, density=True, alpha=0.6, color="C0", edgecolor="k", label="Pseudo-residuals")

        # kde for residuals
        try:
            sns.kdeplot(vals, ax=ax, color="C0", lw=1.5)
        except Exception:
            pass

        # standard normal overlay
        x_min, x_max = np.min(vals), np.max(vals)
        xpad = max(1.0, (x_max - x_min) * 0.1)
        x = np.linspace(x_min - xpad, x_max + xpad, 300)
        ax.plot(x, stats.norm.pdf(x, 0, 1), "k--", lw=1.5, label="Standard normal PDF")

        # add vertical lines for mean and std
        ax.axvline(vals.mean(), color="C1", linestyle=":", lw=1.5, label=f"mean={vals.mean():.2f}")
        ax.axvline(vals.mean() + vals.std(), color="C1", linestyle="--", lw=1)
        ax.axvline(vals.mean() - vals.std(), color="C1", linestyle="--", lw=1)

        ax.set_title(title)
        ax.set_xlabel("Pseudo-residual")
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    return fig


def qq_envelope(residuals, ax):
    """Draw a normal QQ envelope for pseudo-residual draws on an Axes.

    Args:
        residuals: Array with observations on the first axis and draws on the
            second axis.
        ax: Matplotlib Axes on which to draw the envelope and reference line.

    Returns:
        None
    """
    n_obs = residuals.shape[0]

    p = (np.arange(1, n_obs + 1) - 0.5) / n_obs
    theo = stats.norm.ppf(p)

    sorted_res = np.sort(residuals, axis=0)

    lower = np.percentile(sorted_res, 2.5, axis=1)
    mean = np.mean(sorted_res, axis=1)
    upper = np.percentile(sorted_res, 97.5, axis=1)

    ax.fill_between(theo, lower, upper, alpha=0.3)
    ax.plot(theo, mean, lw=2)

    ax.plot(theo, theo, "k--")


def psres_autocorr_plot(
        speed_fpres, 
        load_fpres, 
        angle_fpres, 
        track_id, 
        segment_id, 
        figsize=(18, 6)):
    """Plot autocorrelation summaries for forecast pseudo-residual draws.

    Args:
        speed_fpres: Speed pseudo-residual array with shape
            ``(n_observations, n_draws)``.
        load_fpres: Load pseudo-residual array with shape
            ``(n_observations, n_draws)``.
        angle_fpres: Angle pseudo-residual array with shape
            ``(n_observations, n_draws)``.
        track_id: Track identifier for each observation.
        segment_id: Segment identifier for each observation.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing autocorrelation summaries
        and 95% percentile intervals for the three variables.
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Plot credible intervals of posterior predictive samples
    T, n_draws = speed_fpres.shape

    for i, (ax, fpres, title) in enumerate(zip(
        axs, 
        [speed_fpres, load_fpres, angle_fpres],
        ["Speed", "Accel'rate", "Turn angle"]
    )):

        acfs = []

        lags = 30

        for d in range(n_draws):
            df = pd.DataFrame({
                "track_id": track_id,
                "segment_id": segment_id,
                "fpres": fpres[:, d]
            })
            acf_sample = segmented_acf(df, value_col="fpres", lags=lags)
            acfs.append(acf_sample)

        print("acfs shape:", np.array(acfs).shape)
        mean_acf = np.mean(acfs, axis=0)
        lower_acf = np.percentile(acfs, 2.5, axis=0)
        upper_acf = np.percentile(acfs, 97.5, axis=0)

        ax.plot(np.arange(lags+1), mean_acf, marker='.', color="blue", lw=2, label="Mean ACF (Posterior predictive)")
        ax.fill_between(np.arange(lags+1), lower_acf, upper_acf, color="blue", alpha=0.2, label="95\\% CI (Posterior predictive)")

        # Approximate 95% white-noise limits for the ACF.
        n_observations = len(df)
        wn_limit = norm.ppf(0.975) / np.sqrt(n_observations)
        ax.axhline(wn_limit, color="black", linestyle="--", lw=1, label="95\\% CI (White noise)")
        ax.axhline(-wn_limit, color="black", linestyle="--", lw=1)

    
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        if i == 0:
            legend = ax.legend(
                fancybox=False,
                frameon=True,
                framealpha=1,
                facecolor="white",
            )
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.5)
        ax.set_title(title)

    fig.tight_layout()

    return fig


def plot_transition_probs(cov, tr_matrix, figsize=(18, 10), plot_ci=True):
    """Plot transition probabilities against a covariate for each source state.

    Args:
        cov: One-dimensional covariate values corresponding to the transition
            probabilities.
        tr_matrix: Transition probability samples with shape
            ``(n_states, n_states, n_covariates, n_draws)``. The first two
            dimensions index the source and destination states.
        figsize: Figure size as ``(width, height)`` in inches.
        plot_ci: Whether to show the 95% percentile interval across draws.

    Returns:
        matplotlib.figure.Figure: Figure containing one subplot per source
        state.
    """
    n_states = tr_matrix.shape[0]

    continuous_cmap = plt.get_cmap("plasma")
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(continuous_cmap)(np.linspace(0,1,n_states+1)), N=n_states)

    mosaic = """AABBCC
                .DDEE."""
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=figsize)

    for s1, ax in enumerate(ax_dict.values()):

        probs = tr_matrix[s1,:,:,:]

        for s2 in range(n_states):

            ax.plot(
                cov,
                probs[s2].mean(axis=1),
                linewidth=1,
                color=cmap(s2),
                label=f"State {s2+1}"
                )

            if plot_ci:
                #plot credible intervals of each state
                ax.fill_between(
                    cov,
                    np.percentile(probs[s2], 2.5, axis=1),
                    np.percentile(probs[s2], 97.5, axis=1),
                    color=cmap(s2),
                    alpha=0.3
                )
            
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(cov.min(), cov.max())
        ax.set_title(f"State {s1+1}")
        ax.set_xlabel("Cumulative 30 second Accel'rate")
        ax.set_ylabel("Transition probability")
        if s1 == 0:
            legend = ax.legend(
                fancybox=False,
                frameon=True,
                framealpha=1,
                facecolor="white",
            )
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    return fig


def visualize_emission(scipy_dist, 
                       posterior, 
                       dist_params, 
                       n_states, 
                       xlabel, 
                       xlim, 
                       ylim, 
                       title,
                       plot_ci=True, 
                       ax=None): 
    """Plot state-dependent emission densities on a Matplotlib Axes.

    Args:
        scipy_dist: Distribution to plot. Supported values are ``"gamma"``
            and ``"von_mises"``.
        posterior: Xarray dataset containing posterior parameter samples.
        dist_params: Names of the two posterior parameters used by the
            selected distribution.
        n_states: Number of states represented in the posterior samples.
        xlabel: Label for the x-axis.
        xlim: Lower and upper x-axis limits.
        ylim: Lower and upper y-axis limits.
        title: Subplot title.
        plot_ci: Whether to show the 95% percentile interval across posterior
            samples.
        ax: Axes on which to draw. If ``None``, use the current Axes.

    Raises:
        ValueError: If ``scipy_dist`` is not supported.

    Returns:
        None
    """

    continuous_cmap = plt.get_cmap("plasma")
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(continuous_cmap)(np.linspace(0,1,n_states+1)), N=n_states)

    if ax is None:
        ax = plt.gca()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    if scipy_dist == "gamma":
        stepgrid = np.linspace(xlim[0], xlim[1]+1, 1000)
    elif scipy_dist == "von_mises":
        stepgrid = np.linspace(-np.pi, np.pi, 1000)
    else:
        raise ValueError("Unsupported distribution")
    
    #Posterior is shape (n_states, samples)
    for s in range(n_states):
        print(f"Posterior is shape: {posterior[dist_params[0]].shape}")
        par1 = posterior[dist_params[0]][s,:].values[:, None]
        par2 = posterior[dist_params[1]][s,:].values[:, None]
        
        if scipy_dist == "gamma":
            densities = gamma.pdf(stepgrid, a=par1, scale=1/par2)
            
        elif scipy_dist == "von_mises":
            densities = vonmises.pdf(stepgrid, kappa=par1, loc=par2)
        else:
            raise ValueError("Unsupported distribution")
        
        # Clip densities to avoid extreme values
        densities = np.clip(densities, 0, 100)
        
        mean_density = np.mean(densities, axis=0)
        
        #plot mean densities of each state
        ax.plot(
            stepgrid,
            mean_density,
            linewidth=1,
            color=cmap(s),
            label=f"State {s+1}"
        )
        
        if plot_ci:
            #plot credible intervals of each state
            ax.fill_between(
            stepgrid,
            np.percentile(densities, 2.5, axis=0),
            np.percentile(densities, 97.5, axis=0),
            color=cmap(s),
            alpha=0.3
        )

        ax.set_title(title)
        if title == "Turn angle":
            ax.set_xticks(
                ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                labels=["-180", "-90", "0", "90", "180"])


def plot_emissions_subplot(
    configs,
    n_states,
    figsize=(24, 8),
    plot_ci=True
):
    """Plot configured emission distributions in separate subplots.

    Args:
        configs: Sequence of mappings containing ``scipy_dist``, ``posterior``,
            ``dist_params``, ``xlabel``, ``xlim``, ``ylim``, and ``title`` for
            each subplot.
        n_states: Number of states represented in each distribution.
        figsize: Figure size as ``(width, height)`` in inches.
        plot_ci: Whether to show 95% percentile intervals.

    Returns:
        matplotlib.figure.Figure: Figure containing the emission plots.
    """

    n_plots = len(configs)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axs = [axs]

    for i, (ax, cfg) in enumerate(zip(axs, configs)):
        visualize_emission(
            scipy_dist=cfg["scipy_dist"],
            posterior=cfg["posterior"],
            dist_params=cfg["dist_params"],
            n_states=n_states,
            xlabel=cfg["xlabel"],
            xlim=cfg["xlim"],
            ylim=cfg["ylim"],
            title=cfg["title"],
            plot_ci=plot_ci,
            ax=ax
        )

        if i == 0:
            legend = ax.legend(
                fancybox=False,
                frameon=True,
                framealpha=1,
                facecolor="white",
            )
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    return fig


def plot_pp_density(speed, 
                    accelrate, 
                    turn_angle,
                    pp_samples_speed,
                    pp_samples_accelrate,
                    pp_samples_turn_angle,
                    figsize=(18, 6)):
    """Compare observed densities with posterior predictive densities.

    Each predictive draw is used to estimate a density; the plot shows the
    mean predictive density and its 95% percentile interval.

    Args:
        speed: One-dimensional observed speed values.
        accelrate: One-dimensional observed acceleration-rate values.
        turn_angle: One-dimensional observed turn-angle values in radians.
        pp_samples_speed: Iterable of one-dimensional posterior predictive
            speed samples.
        pp_samples_accelrate: Iterable of one-dimensional posterior
            predictive acceleration-rate samples.
        pp_samples_turn_angle: Iterable of one-dimensional posterior
            predictive turn-angle samples.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing the three density plots.
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=False)

    for i, (ax, obs, pp_samples, title, xlabel) in enumerate(zip(
        axs, 
        [speed, accelrate, turn_angle], 
        [pp_samples_speed, pp_samples_accelrate, pp_samples_turn_angle], 
        ["Speed", "Accel'Rate", "Turn angle"],
        ["Meters per second (m/s)", "Arbitrary units (AU)", "Degrees ($^\\circ$)"]
    )):

        sns.kdeplot(obs, color="red", alpha=1, lw=2, ax=ax, label="Density of observations")

        x = ax.lines[-1].get_xdata()
        densities = np.array([
            gaussian_kde(sample)(x)
            for sample in pp_samples
        ])

        mean_density = densities.mean(axis=0)
        lower_density = np.percentile(densities, 2.5, axis=0)
        upper_density = np.percentile(densities, 97.5, axis=0)

        ax.plot(x, mean_density, color="blue", lw=2, label="Mean predictive density")
        ax.fill_between(x, lower_density, upper_density, color="blue", alpha=0.2, label="95\\% CI of predictive density")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        if title == "Turn angle":
            ax.set_xticks(
                ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                labels=["-180", "-90", "0", "90", "180"])
        ax.set_title(title)
        if i == 0:
            legend = ax.legend(
                fancybox=False,
                frameon=True,
                framealpha=1,
                facecolor="white",
            )
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.5)
    
    fig.tight_layout()
    return fig


def circular_acf(x, k=1):
    """Compute the circular autocorrelation at a specified lag.

    Args:
        x: One-dimensional circular data in radians.
        k: Non-negative integer lag.

    Returns:
        float: Circular autocorrelation coefficient.

    Raises:
        ValueError: If ``k`` is greater than or equal to the data length.
    """
    T = len(x)
    if k >= T:
        raise ValueError("Lag must be less than the length of the data.")

    if k == 0:
        return 1.0  # Autocorrelation at lag 0 is always 1

    muhat = circmean(x, low=-np.pi, high=np.pi)

    return np.sum(np.sin(x - muhat)[:T-k] * np.sin(x - muhat)[k:]) / np.sum(np.sin(x - muhat)**2)


def segmented_acf(df, value_col="speed", lags=100):
    """Compute a pair-count-weighted ACF across track segments.

    Segments with ``lags`` or fewer observations are excluded. Turn-angle
    values use circular autocorrelation; other values use the statsmodels ACF.

    Args:
        df: DataFrame containing ``track_id``, ``segment_id``, and the column
            named by ``value_col``.
        value_col: Name of the data column to use for the autocorrelation.
        lags: Maximum lag to compute.

    Returns:
        numpy.ndarray: Weighted average autocorrelation for lags zero through
        ``lags``.
    """
    acfs = []
    weights = []

    for _, seg in df.groupby(["track_id", "segment_id"]):
        x = seg[value_col].to_numpy()

        # Need at least lags+1 observations
        if len(x) <= lags:
            continue

        if value_col == "turn_angle":
            # Use circular autocorrelation for turn angles
            acf_seg = [circular_acf(x, lag) for lag in range(lags + 1)]
        else:
            acf_seg = acf(x, nlags=lags, fft=True)

        acfs.append(acf_seg)

        # Number of observation pairs for each lag
        weights.append(len(x) - np.arange(lags + 1))

    acfs = np.asarray(acfs)
    weights = np.asarray(weights)

    return np.average(acfs, axis=0, weights=weights)


def plot_autocorr(
        speed, 
        accelrate, 
        turn_angle,
        track_id,
        segment_id, 
        pp_samples_speed,
        pp_samples_accelrate,
        pp_samples_turn_angle,
        lags=100,
        figsize=(24, 6)
    ):
    """Plot observed and posterior predictive autocorrelations.

    Args:
        speed: One-dimensional observed speed values.
        accelrate: One-dimensional observed acceleration-rate values.
        turn_angle: One-dimensional observed turn-angle values in radians.
        track_id: Track identifier for each observation.
        segment_id: Segment identifier for each observation.
        pp_samples_speed: Posterior predictive speed values with shape
            ``(n_draws, n_observations)``.
        pp_samples_accelrate: Posterior predictive acceleration-rate values
            with shape ``(n_draws, n_observations)``.
        pp_samples_turn_angle: Posterior predictive turn-angle values with
            shape ``(n_draws, n_observations)``.
        lags: Maximum lag to plot.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing the three ACF plots and
        predictive intervals.
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for i, (ax, obs, pp_samples, col, title) in enumerate(zip(
        axs, 
        [speed, accelrate, turn_angle], 
        [pp_samples_speed, pp_samples_accelrate, pp_samples_turn_angle], 
        ["speed", "accelrate", "turn_angle"],
        ["Speed", "Accel'Rate", "Turn angle"]
    )):
        df = pd.DataFrame({
            "track_id": track_id,
            "segment_id": segment_id,
            col: obs
        })

        acf_vals = segmented_acf(df, value_col=col, lags=lags)
        ax.plot(np.arange(lags+1), acf_vals, marker='.', color="blue", lw=2, label="Observed data")

        # Plot credible intervals of posterior predictive samples
        n_draws = pp_samples.shape[0]

        pp_sample_acfs = []

        for d in range(n_draws):
            pp_samples_df = pd.DataFrame({
                "track_id": track_id,
                "segment_id": segment_id,
                col: pp_samples[d]
            })

            acf_sample = segmented_acf(pp_samples_df, value_col=col, lags=lags)
            pp_sample_acfs.append(acf_sample)

        lower = np.percentile(pp_sample_acfs, 2.5, axis=0)
        upper = np.percentile(pp_sample_acfs, 97.5, axis=0)

        ax.fill_between(np.arange(lags+1), lower, upper, color="blue", alpha=0.2, label="95\\% CI (Posterior predictive)")
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        if i == 0:
            legend = ax.legend(
                fancybox=False,
                frameon=True,
                framealpha=1,
                facecolor="white",
            )
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()

    return fig


def plot_autocorr_subplots(obs, ppspeeds, pploads, ppangles, figsize=(7, 21)):
    """Create a figure for autocorrelation plots of three data variables.

    Args:
        obs: Observed data supplied to the plotting routine.
        ppspeeds: Posterior predictive speed samples.
        pploads: Posterior predictive acceleration-rate samples.
        ppangles: Posterior predictive turn-angle samples.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: Figure created for the autocorrelation
        subplots.
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    plot_autocorr(obs, ppspeeds, lags=30, value_col="speed", title="Speed", ax=axs[0])
    plot_autocorr(obs, pploads, lags=30, value_col="accelrate", title="Accel'Rate", ax=axs[1])
    plot_autocorr(obs, ppangles, lags=30, value_col="turn_angle", title="Turn angle", ax=axs[2])

    # Common title

    return fig


def plot_viterbi(obs, viterbi, n_states, figsize=(18,18)):
    """Plot randomly selected decoded movement paths colored by state.

    Args:
        obs: DataFrame containing ``x``, ``y``, ``segment_id``, ``player_id``,
            and ``track_id`` columns.
        viterbi: Decoded state sequence aligned with the rows of ``obs``.
            State labels are expected to be integers from 1 through
            ``n_states``.
        n_states: Number of hidden states.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing up to nine sampled paths.
    """

    continuous_cmap = plt.get_cmap("plasma")
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(continuous_cmap)(np.linspace(0,1,n_states+1)), N=n_states)
    fig, axs = plt.subplots(3, 3, figsize=figsize)
    path_len = 30

    for j, ax in enumerate(axs.flatten()):
        # Ensure start_idx leaves room for a full path_len slice
        max_start = obs.shape[0] - path_len
        if max_start <= 0:
            continue
        start_idx = np.random.randint(max_start)

        x = obs["x"][start_idx : start_idx+path_len].to_numpy()
        y = obs["y"][start_idx : start_idx+path_len].to_numpy()
        c = viterbi[start_idx : start_idx+path_len]
        seg = obs["segment_id"][start_idx : start_idx+path_len].to_numpy()

        # Skip if slice is too short to plot
        if len(x) < 2:
            continue

        for state in range(1, n_states + 1):
            ax.plot([], [], color=cmap(state - 1), label=f"State {state}")

        for i in range(len(x) - 1):
            if seg[i] != seg[i+1]:
                break
            ax.plot(x[i:i+2], y[i:i+2], color=cmap((c[i] - 1)/ (n_states - 1)), linewidth=2)
            ax.scatter(x[i:i+2], y[i:i+2], color=cmap((c[i] - 1)/ (n_states - 1)))
            if i == 0:
                ax.annotate("start", (x[i]+0.2, y[i]+0.2))
        
        # Set a title for the subplot indicating player id and track id
        player_id = obs["player_id"][start_idx]
        track_id = obs["track_id"][start_idx]

        ax.set_title(f"Player: {player_id}, Track: {track_id}")
        ax.set_aspect("equal", adjustable="datalim")

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, color="lightgray", linewidth=0.8, alpha=0.8)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if j == 0:
            legend = ax.legend(
                fancybox=False,
                frameon=True,
                framealpha=1,
                facecolor="white",
            )
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()

    return fig


def plot_state_proportions(track_id, segment_id, player_id, game, viterbi, timestamp, n_states, figsize=(21,7)):
    """Plot cumulative state proportions over time for sampled tracks.

    Creates a 2-by-3 grid, with rows for segments and columns for up to three
    randomly sampled tracks. Each subplot shows the cumulative proportion of
    observations assigned to each state.

    Args:
        track_id: Track identifier for each observation.
        segment_id: Segment identifier for each observation; values are used
            as row indices in the subplot grid.
        player_id: Player identifier for each observation, used in subplot
            titles.
        game: Game identifier for each observation, used in subplot titles.
        viterbi: Decoded state sequence aligned with the other inputs. State
            labels are expected to be integers from 1 through ``n_states``.
        timestamp: Timestamp for each observation.
        n_states: Number of hidden states, determining the number of lines in
            each subplot.
        figsize: Figure size as ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing the state proportion
        plots for up to three randomly sampled tracks.
    """
    fig, axs = plt.subplots(2, 3, figsize=figsize)

    continuous_cmap = plt.get_cmap("plasma")
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(continuous_cmap)(np.linspace(0,1,n_states+1)), N=n_states)

    df = pd.DataFrame({
        "timestamp": timestamp,
        "track_id": track_id,
        "segment_id": segment_id,
        "game": game,
        "viterbi": viterbi,
        "player_id": player_id
    })

    tracks = df["track_id"].unique()
    random_tracks = np.random.choice(tracks, size=np.min([len(tracks), 3]), replace=False)

    # First values do not add much value to the plot
    start = 15
    
    for j in range(axs.shape[1]):
        for i in range(axs.shape[0]):

            if j >= len(random_tracks): return fig
            track = random_tracks[j]
            df_track = df[(df["track_id"] == track) & (df["segment_id"] == i)].reset_index(drop=True)

            for state in range(1,n_states+1):
                cumulative_count = np.cumsum(df_track["viterbi"] == state) / np.arange(1, df_track.shape[0]+1)
                axs[i, j].plot(df_track["timestamp"][start:], cumulative_count[start:], color=cmap(state-1), linewidth=1, label=f"State {state}")
                #ax.fill_between(df_track["timestamp"], cumulative_count, color=cmap(state-1), alpha=0.1)

            axs[i, j].set_xlabel("Timestamp")
            axs[i, j].set_ylabel("Proportion")
            axs[i, j].set_title(f"Player: {df_track["player_id"][0]}, Match: {df_track["game"][0]}, Half: {df_track["segment_id"][0] + 1}")
            #ax.set_xlim(timestamp.min(), timestamp.max())
            axs[i, j].set_ylim(0, 1)

            # set x-ticks to 10 minute intervals and to form hh:mm. Timestamp is datetime64[ns] type, so we can use pandas to format the ticks
            axs[i, j].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            axs[i, j].xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
            axs[i, j].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axs[i, j].tick_params(axis='x', rotation=45)

            if i == 0 and j == 0:
                legend = axs[i, j].legend(
                    fancybox=False,
                    frameon=True,
                    framealpha=1,
                    facecolor="white",
                )
                legend.get_frame().set_edgecolor("black")
                legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    
    return fig
    

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    """Run the placeholder plotting command.

    Args:
        input_path: Path to the input CSV. Currently unused by this placeholder
            command.
        output_path: Path for the output plot. Currently unused by this
            placeholder command.

    Returns:
        None
    """
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
