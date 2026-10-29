from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import vonmises, norm
from scipy.special import softmax
import scipy.stats as stats
import pandas as pd
import arviz as az

from src.soccer_hmm.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def build_transition_matrix(betas, design_matrix, n_states):
    """Build covariate-dependent transition matrices from posterior draws.

    The transition model uses one baseline logit per previous state. The
    baseline is kept at zero and the remaining logits are computed from the
    corresponding regression coefficients before applying a softmax.

    Args:
        betas: Regression coefficients with shape
            ``(n_states * (n_states - 1), n_covariates, n_draws)``.
        design_matrix: Covariate matrix with shape ``(n_covariates, T)``.
        n_states: Number of hidden states.

    Returns:
        An array of transition probabilities with shape
        ``(n_states, n_states, T, n_draws)``. The first state axis is the
        previous state and the second is the next state.
    """

    T = design_matrix.shape[1]
    n_draws = betas.shape[2]

    tr_matrix = np.zeros((n_states, n_states, T, n_draws))

    row = 0
    for i in range(n_states):
        logits = np.zeros((n_states, T, n_draws))  # includes baseline
        col_idx = [j for j in range(n_states) if j != i]

        for j in col_idx:
            logits[j] = (betas[row].T @ design_matrix).T
            row += 1

        # The omitted destination state is the baseline with logit zero.
        tr_matrix[i] = softmax(logits, axis=0)

    assert np.allclose(tr_matrix.sum(axis=1), 1.0, atol=1e-8)

    return tr_matrix

def extract_draws(fit, combined=True):
    """Extract posterior samples from a CmdStanPy fit.

    Args:
        fit: A CmdStanPy fit object accepted by ``arviz.from_cmdstanpy``.
        combined: If True, combine chains into one draw dimension. If False,
            preserve separate chain and draw dimensions.

    Returns:
        An xarray object containing the extracted posterior samples.
    """
    idata = az.from_cmdstanpy(posterior=fit)
    posterior_draws = az.extract(idata, combined=combined)
    return posterior_draws


def compute_allprobs(speed, accelrate, turn_angle, shape_speeds, rate_speeds, shape_loads, rate_loads, loc, kappa):
    """Compute joint emission densities for all observations and draws.

    The three observed variables are conditionally independent given the
    hidden state, so their state-specific densities are multiplied together.
    Broadcasting evaluates all time points, states, and posterior draws at
    once.

    Args:
        speed: Speed observations with shape ``(T,)``.
        accelrate: Acceleration-rate observations with shape ``(T,)``.
        turn_angle: Turn-angle observations with shape ``(T,)``.
        shape_speeds: Gamma shape parameters with shape ``(n_states, n_draws)``.
        rate_speeds: Gamma rate parameters with shape ``(n_states, n_draws)``.
        shape_loads: Gamma shape parameters for acceleration rate with shape
            ``(n_states, n_draws)``.
        rate_loads: Gamma rate parameters for acceleration rate with shape
            ``(n_states, n_draws)``.
        loc: Von Mises location parameters with shape ``(n_states, n_draws)``.
        kappa: Von Mises concentration parameters with shape
            ``(n_states, n_draws)``.

    Returns:
        An array of joint emission densities with shape ``(T, n_states,
        n_draws)``.
    """
    #print("computing emission probabilities for all draws (vectorized)...")

    # broadcastable shapes: (T,1,1) vs (1,n_states,n_draws)
    x_speed = speed[:, None, None]
    x_accelrate = accelrate[:, None, None]
    x_turn = turn_angle[:, None, None]

    ss = shape_speeds[None, :, :]
    rs = rate_speeds[None, :, :]
    sl = shape_loads[None, :, :]
    rl = rate_loads[None, :, :]
    locs = loc[None, :, :]
    kappas = kappa[None, :, :]

    # PDFs computed via broadcasting -> shape (T, n_states, n_draws)
    p_speed = gamma_dist.pdf(x_speed, a=ss, scale=1.0 / rs)
    p_pl = gamma_dist.pdf(x_accelrate, a=sl, scale=1.0 / rl)
    p_turn = vonmises.pdf(x_turn, kappa=kappas, loc=locs)

    allprobs = p_speed * p_pl * p_turn  # shape (T, n_states, n_draws)

    return allprobs

def hmm_lalpha(allprobs, tr_matrix, delta):
    """Compute scaled log-forward probabilities for all posterior draws.

    The recursion is sequential in time but vectorized over posterior draws.
    Scaling at each time step prevents underflow while ``lscale`` restores the
    accumulated log probability in the returned values.

    Args:
        allprobs: Emission densities with shape ``(T, n_states, n_draws)``.
        tr_matrix: Transition probabilities with shape
            ``(n_states, n_states, T, n_draws)``.
        delta: Initial state probabilities with shape ``(n_states,)``.

    Returns:
        Log-forward probabilities with shape ``(n_states, T, n_draws)``.
    """
    T, n_states, n_draws = allprobs.shape

    lalpha = np.full((n_states, T, n_draws), np.nan)

    # Keep draws as columns so the recursion can process every draw together.
    foo = delta[:, None] * allprobs[0, :, :]
    sumfoo = np.sum(foo, axis=0)

    lscale = np.log(sumfoo)
    foo /= sumfoo[None, :]
    lalpha[:, 0, :] = np.log(foo) + lscale[None, :]

    for t in range(1, T):
        foo = np.einsum("sd,srd->rd", foo, tr_matrix[:, :, t, :])
        foo *= allprobs[t, :, :]

        sumfoo = np.sum(foo, axis=0)

        lscale += np.log(sumfoo)
        foo /= sumfoo[None, :]
        lalpha[:, t, :] = np.log(foo) + lscale[None, :]
            
    return lalpha


def hmm_pseudo_residuals(x, cdf_fun, cdf_params,
                           allprobs, tr_matrix):
    """Compute forecast pseudo-residuals for an HMM observation sequence.

    For each observation, the state-specific CDF values are mixed using the
    one-step-ahead state probabilities. The resulting probability is mapped
    to a standard normal scale with the inverse normal CDF.

    Args:
        x: Observations with shape ``(T,)``.
        cdf_fun: Callable that accepts an observation array and a state
            parameter dictionary and returns CDF values.
        cdf_params: State-specific parameter dictionaries. The list length
            must equal ``n_states``.
        allprobs: Emission densities with shape ``(T, n_states, n_draws)``.
        tr_matrix: Transition probabilities with shape
            ``(n_states, n_states, T, n_draws)``.

    Returns:
        Forecast pseudo-residuals with shape ``(T, n_draws)``. Missing input
        observations remain ``NaN``.
    """
    T = len(x)
    n_states = len(cdf_params)

    _, _, _, n_draws = tr_matrix.shape

    delta = np.repeat(1.0 / n_states, n_states)

    la = hmm_lalpha(allprobs, tr_matrix, delta)

    # pstep[t, state, draw] is the state-specific CDF at observation t.
    pstep = np.full((T, n_states, n_draws), np.nan)

    # One residual is produced for every observed time and posterior draw.
    fpres = np.full((T, n_draws), np.nan)

    for state, params in enumerate(cdf_params):
        pstep[:, state, :] = cdf_fun(x[:, None], params)

    if not np.isnan(x[0]):
        fpres[0, :] = norm.ppf(delta @ pstep[0])

    # The state prediction is recursive in time but vectorized over draws.
    for t in range(1, T):
        alpha = np.exp(la[:, t - 1, :] - np.max(la[:, t - 1, :], axis=0))
        pred = np.einsum("sd,srd->rd", alpha, tr_matrix[:, :, t, :])
        pred /= np.sum(alpha, axis=0)
        prob = np.einsum("sd,sd->d", pred, pstep[t, :, :])

        fpres[t, :] = norm.ppf(prob)

    return fpres


def gamma_cdf(x, params):
    """Evaluate a Gamma CDF using shape and rate parameterization.

    Args:
        x: Values at which to evaluate the CDF. May be scalar or array-like.
        params: Mapping with ``"shape"`` and ``"rate"`` entries.

    Returns:
        Gamma CDF values with the broadcasted shape of ``x`` and the
        parameters.
    """
    return gamma_dist.cdf(
        x,
        a=params["shape"],
        scale=1.0 / params["rate"]
    )


def vonmises_cdf(x, params, eps=1e-12):
    """Evaluate a wrapped von Mises CDF for angular observations.

    Args:
        x: Angles in radians. May be scalar or array-like.
        params: Mapping with ``"loc"`` and ``"kappa"`` entries.
        eps: Lower and upper clipping distance from zero and one.

    Returns:
        Clipped von Mises CDF values with the broadcasted shape of ``x`` and
        the parameters.
    """
    delta = x - params["loc"] # compute the difference between the observed angle and the mean direction
    delta_wrapped = np.arctan2(np.sin(delta), np.cos(delta)) # wrap the difference to the range [-pi, pi]
    val = vonmises.cdf(delta_wrapped, kappa=params["kappa"], loc=0) # compute the CDF of the wrapped difference using the von Mises distribution
    return np.clip(val, eps, 1 - eps)


def compute_all_pseudo_residuals(speed, accelrate, turn_angle, shape_speeds, rate_speeds, shape_loads, rate_loads, loc, kappa, tr_matrix, n_states):
    """Compute pseudo-residuals for all three observed variables.

    Args:
        speed: Speed observations with shape ``(T,)``.
        accelrate: Acceleration-rate observations with shape ``(T,)``.
        turn_angle: Turn-angle observations with shape ``(T,)``.
        shape_speeds: Speed Gamma shape parameters, shape ``(n_states, n_draws)``.
        rate_speeds: Speed Gamma rate parameters, shape ``(n_states, n_draws)``.
        shape_loads: Acceleration-rate Gamma shape parameters.
        rate_loads: Acceleration-rate Gamma rate parameters.
        loc: Turn-angle von Mises locations.
        kappa: Turn-angle von Mises concentrations.
        tr_matrix: Transition probabilities with shape
            ``(n_states, n_states, T, n_draws)``.
        n_states: Number of hidden states.

    Returns:
        A tuple ``(speed_res, load_res, angle_res)``. Each array has shape
        ``(T, n_draws)``.
    """

    allprobs = compute_allprobs(speed, accelrate, turn_angle, shape_speeds, rate_speeds, shape_loads, rate_loads, loc, kappa)

    # Speed residuals, shape_speeds and rate_speeds are now arrays of shape (n_states, n_draws)
    speed_params = [
        {"shape": shape_speeds[j], "rate": rate_speeds[j]}
        for j in range(n_states)
    ]

    speed_res = hmm_pseudo_residuals(
        speed, gamma_cdf, speed_params,
        allprobs, tr_matrix
    )

    # Playerload residuals
    load_params = [
        {"shape": shape_loads[j], "rate": rate_loads[j]}
        for j in range(n_states)
    ]

    load_res = hmm_pseudo_residuals(
        accelrate, gamma_cdf, load_params,
        allprobs, tr_matrix
    )

    # Turn angle residuals
    angle_params = [
        {"loc": loc[j], "kappa": kappa[j]}
        for j in range(n_states)
    ]

    angle_res = hmm_pseudo_residuals(
        turn_angle, vonmises_cdf, angle_params,
        allprobs, tr_matrix
    )
    
    return speed_res, load_res, angle_res


def simulate_from_posterior_predictive(
        T, segment_id, track_id,
        shape_speeds, rate_speeds, shape_loads, rate_loads,
        loc_angles, kappa_angles, beta,
        global_mean, global_sd,
        rng,
        window_samples=15,
    ):
    """Simulate observations and hidden states from posterior draws.

        Draws are simulated in parallel at each time step. The time loop remains
        because the next state and acceleration-rate covariate depend on the
        previous simulated values.

        Args:
            T: Number of time points to simulate.
            segment_id: Segment identifier for each time point, shape ``(T,)``.
            track_id: Track identifier for each time point, shape ``(T,)``.
            shape_speeds: Speed Gamma shape parameters, shape ``(n_states, n_draws)``.
            rate_speeds: Speed Gamma rate parameters, shape ``(n_states, n_draws)``.
            shape_loads: Acceleration-rate Gamma shape parameters.
            rate_loads: Acceleration-rate Gamma rate parameters.
            loc_angles: Von Mises location parameters, shape ``(n_states, n_draws)``.
            kappa_angles: Von Mises concentration parameters.
            beta: Transition coefficients with shape
                ``(n_states * (n_states - 1), 2, n_draws)``.
            global_mean: Mean used to standardize the cumulative load covariate.
            global_sd: Standard deviation used to standardize the cumulative load
                covariate.
            rng: NumPy random number generator.
            window_samples: Number of previous samples included in the cumulative
                load covariate.

        Returns:
            A tuple ``(ppspeeds, pploads, ppangles, ppstates)``. The simulated
            continuous arrays have shape ``(n_draws, T)`` and ``ppstates`` has the
            same shape with integer state indices.
    """

    n_states, n_draws = shape_speeds.shape
    ppstates = np.zeros((n_draws, T), dtype=int)
    ppspeeds = np.full((n_draws, T), np.nan)
    pploads = np.full((n_draws, T), np.nan)
    ppangles = np.full((n_draws, T), np.nan)

    segment_id = np.asarray(segment_id)
    track_id = np.asarray(track_id)

    # A new segment resets the state distribution and load history.
    new_segment = np.r_[True, (segment_id[1:] != segment_id[:-1]) |
                        (track_id[1:] != track_id[:-1])]
    segment_start = np.maximum.accumulate(np.where(new_segment, np.arange(T), 0))

    init_probs = np.repeat(1 / n_states, n_states)
    draw_idx = np.arange(n_draws)
    ppstates[:, 0] = rng.choice(n_states, size=n_draws, p=init_probs)
    ppspeeds[:, 0] = gamma_dist.rvs(
        a=shape_speeds[ppstates[:, 0], draw_idx],
        scale=1.0 / rate_speeds[ppstates[:, 0], draw_idx],
        random_state=rng,
    )
    pploads[:, 0] = gamma_dist.rvs(
        a=shape_loads[ppstates[:, 0], draw_idx],
        scale=1.0 / rate_loads[ppstates[:, 0], draw_idx],
        random_state=rng,
    )
    ppangles[:, 0] = vonmises.rvs(
        loc=loc_angles[ppstates[:, 0], draw_idx],
        kappa=kappa_angles[ppstates[:, 0], draw_idx],
        random_state=rng,
    )

    # Group coefficients by previous state for direct vectorized lookup.
    beta_by_prev_state = beta.reshape(n_states, n_states - 1, 2, n_draws)
    cumulative_load = np.cumsum(pploads, axis=1)

    for t in range(1, T):
        if new_segment[t]:
            ppstates[:, t] = rng.choice(n_states, size=n_draws, p=init_probs)
        else:
            window_start = max(t - window_samples, segment_start[t])
            raw_cumulative_ar = cumulative_load[:, t - 1] - cumulative_load[:, window_start - 1] if window_start else cumulative_load[:, t - 1]
            cumulative_ar_z = (raw_cumulative_ar - global_mean) / global_sd
            cov_t = np.column_stack((np.ones(n_draws), cumulative_ar_z))

            prev_state = ppstates[:, t - 1]
            selected_beta = beta_by_prev_state[prev_state, :, :, draw_idx]
            logits = np.zeros((n_draws, n_states))
            beta_logits = np.einsum("dsc,dc->ds", selected_beta, cov_t)

            # Each previous state has its own omitted baseline destination.
            for state in range(n_states):
                target_draws = prev_state != state
                beta_pos = state - (prev_state < state)
                logits[target_draws, state] = beta_logits[target_draws, beta_pos[target_draws]]

            probs = softmax(logits, axis=1)
            ppstates[:, t] = np.sum(rng.random((n_draws, 1)) > np.cumsum(probs, axis=1), axis=1)

        state_idx = ppstates[:, t]
        ppspeeds[:, t] = gamma_dist.rvs(
            a=shape_speeds[state_idx, draw_idx],
            scale=1.0 / rate_speeds[state_idx, draw_idx],
            random_state=rng,
        )
        pploads[:, t] = gamma_dist.rvs(
            a=shape_loads[state_idx, draw_idx],
            scale=1.0 / rate_loads[state_idx, draw_idx],
            random_state=rng,
        )
        ppangles[:, t] = vonmises.rvs(
            loc=loc_angles[state_idx, draw_idx],
            kappa=kappa_angles[state_idx, draw_idx],
            random_state=rng,
        )
        cumulative_load[:, t] = cumulative_load[:, t - 1] + pploads[:, t]

    return ppspeeds, pploads, ppangles, ppstates

# if your Stan model uses mean/sd -> gamma shape/rate
def musigma_to_gamma(mu, sigma):
    """Convert Gamma mean and standard deviation to shape and rate.

    Args:
        mu: Gamma means.
        sigma: Gamma standard deviations.

    Returns:
        A tuple ``(shape, rate)`` with the same broadcasted shape as ``mu``
        and ``sigma``.
    """
    shape = (mu / sigma) ** 2
    rate = mu / (sigma ** 2)
    return shape, rate


# xangle/yangle -> von Mises loc/kappa
def xy_to_vonmises(xangle, yangle):
    """Convert mean-direction coordinates to von Mises parameters.

    Args:
        xangle: Mean-direction x components.
        yangle: Mean-direction y components.

    Returns:
        A tuple ``(loc, kappa)`` containing the direction in radians and the
        concentration. Both outputs have the broadcasted input shape.
    """
    loc = np.arctan2(yangle, xangle)
    kappa = np.sqrt(xangle**2 + yangle**2)
    return loc, kappa


from scipy.stats import truncnorm

def simulate_ordered_mu_speeds_sequential(n_draws, means, sds, rng):
    """Sample ordered state means independently across posterior draws.

    Each state mean is sampled from a normal distribution truncated below the
    previously sampled state mean, enforcing increasing means within a draw.

    Args:
        n_draws: Number of posterior draws to sample.
        means: Mean of the proposal distribution for each state.
        sds: Standard deviation of the proposal distribution for each state.
        rng: NumPy random number generator.

    Returns:
        An array of ordered means with shape ``(n_states, n_draws)``.
    """
    n_states = len(means)
    result = np.zeros((n_states, n_draws))
    lower = np.zeros(n_draws)

    for s in range(n_states):
        a = (lower - means[s]) / sds[s]   # truncnorm's `a` is in standardized units
        b = np.inf
        result[s, :] = truncnorm.rvs(a, b, loc=means[s], scale=sds[s], size=n_draws, random_state=rng)
        lower = result[s, :]

    return result

def simulate_from_priors(rng, n_draws, n_states):
    """Generate parameter draws from the model's prior distributions.

    Args:
        rng: NumPy random number generator.
        n_draws: Number of prior draws to generate.
        n_states: Number of hidden states.

    Returns:
        A dictionary containing Gamma emission parameters, von Mises emission
        parameters, and transition coefficients. Parameter arrays use shape
        ``(n_states, n_draws)`` except ``beta``, which has shape
        ``(n_states * (n_states - 1), 2, n_draws)``.
    """

    mu_speed_mus = [0.0, 1.3, 1.7, 2.9, 3.1]
    mu_speed_sds = [0.5, 0.5, 0.5, 0.5, 0.5]

    mu_speeds = simulate_ordered_mu_speeds_sequential(
        n_draws, mu_speed_mus, mu_speed_sds, rng  # confirm against Stan
    )

    # mu_loads: normal(5, 4) truncated at 0 -- confirm mean/sd and whether it's ordered too
    mu_load_mus = np.array([0.5, 0.5, 5.0, 9.0, 9.0])
    a = (0.0 - mu_load_mus) / 0.5
    mu_loads = truncnorm.rvs(a, np.inf, loc=mu_load_mus, scale=0.5, size=(n_draws, n_states), random_state=rng).T

    sigma_speeds = np.abs(rng.standard_t(3, (n_states, n_draws)))
    sigma_loads = np.abs(rng.standard_t(3, (n_states, n_draws)))
    xangle = rng.normal(5, 2.5, (n_states, n_draws))
    yangle = rng.normal(0, 0.5, (n_states, n_draws))
    beta = rng.normal(0, 1, (n_states * (n_states - 1), 2, n_draws))

    shape_speeds, rate_speeds = musigma_to_gamma(mu_speeds, sigma_speeds)
    shape_loads, rate_loads = musigma_to_gamma(mu_loads, sigma_loads)
    loc, kappa = xy_to_vonmises(xangle, yangle)

    return {
        "shape_speeds": shape_speeds, "shape_loads": shape_loads,
        "rate_speeds": rate_speeds, "rate_loads": rate_loads,
        "loc": loc, "kappa": kappa, "beta": beta,
    }





@app.command()
def main():
    """Provide the module's Typer entry point."""
    return


if __name__ == "__main__":
    app()
