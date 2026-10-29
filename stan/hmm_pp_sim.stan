data {
    int<lower=0> T;
    array[T] int track_ID;
    array[T] int segment_ID;
    int<lower=1> N;
    int n_covs;
    real global_mean;
    real global_sd;
    int<lower=1> window_samples;
}

parameters {
    // same declarations as the fitted model -- needed so Stan can read
    // the incoming draws' shapes correctly, even though nothing is estimated here
    positive_ordered[N] mu_speeds;
    vector<lower=0>[N] sigma_speeds;
    vector<lower=0>[N] mu_loads;
    vector<lower=0>[N] sigma_loads;
    vector[N] xangle;
    vector[N] yangle;
    matrix[N*(N-1), n_covs+1] beta;
}

transformed parameters {
    // identical block to your main model, recomputes shape/rate/loc/kappa
    // from whatever draws are fed in via generate_quantities
    vector<lower=0>[N] shape_speeds;
    vector<lower=0>[N] shape_loads;
    vector<lower=0>[N] rate_speeds;
    vector<lower=0>[N] rate_loads;
    vector<lower=-pi(),upper=pi()>[N] loc;
    vector<lower=0>[N] kappa;

    for (n in 1:N) {
        loc[n] = atan2(yangle[n], xangle[n]);
        kappa[n] = sqrt(xangle[n]*xangle[n] + yangle[n]*yangle[n]);
        shape_speeds[n] = mu_speeds[n]*mu_speeds[n]/(sigma_speeds[n]*sigma_speeds[n]);
        shape_loads[n]  = mu_loads[n]*mu_loads[n]/(sigma_loads[n]*sigma_loads[n]);
        rate_speeds[n]  = mu_speeds[n]/(sigma_speeds[n]*sigma_speeds[n]);
        rate_loads[n]   = mu_loads[n]/(sigma_loads[n]*sigma_loads[n]);
    }
}

generated quantities {

    // ---- posterior/prior predictive simulation ----
    array[T] int<lower=1,upper=N> pp_states;
    array[T] real<lower=0> pp_speeds;
    array[T] real<lower=0> pp_loads;
    array[T] real<lower=-pi(),upper=pi()> pp_angles;

    {
        array[T] int segment_start;
        int current_start = 1;

        for (t in 1:T) {
            if (t > 1 && (track_ID[t] != track_ID[t-1] || segment_ID[t] != segment_ID[t-1])) {
                current_start = t;
            }
            segment_start[t] = current_start;
        }

        for (t in 1:T) {
            if (t == 1 || track_ID[t] != track_ID[t-1] || segment_ID[t] != segment_ID[t-1]) {
                real raw_angle;
                pp_states[t] = categorical_rng(rep_vector(1.0 / N, N));

                pp_speeds[t] = gamma_rng(shape_speeds[pp_states[t]], rate_speeds[pp_states[t]]);
                pp_loads[t]  = gamma_rng(shape_loads[pp_states[t]], rate_loads[pp_states[t]]);
                raw_angle = von_mises_rng(loc[pp_states[t]], kappa[pp_states[t]]);
                pp_angles[t] = atan2(sin(raw_angle), cos(raw_angle));

            } else {
                int window_start = max(t - window_samples, segment_start[t]);
                real raw_cum = 0;
                for (i in window_start:(t - 1)) {
                    raw_cum += pp_loads[i]; 
                }
                real cum_z = (raw_cum - global_mean) / global_sd;

                int prev = pp_states[t - 1];
                vector[N] eta;
                int betarow = (prev - 1) * (N - 1) + 1;

                for (j in 1:N) {
                    if (j == prev) {
                        eta[j] = 0;  // reference state, matches model block
                    } else {
                        eta[j] = beta[betarow, 1] * 1.0 + beta[betarow, 2] * cum_z;
                        betarow += 1;
                    }
                }

                if (is_nan(sum(eta)) || is_inf(sum(eta))) {
                    reject("eta contains nan/inf at t=", t, " raw_cum=", raw_cum, " cum_z=", cum_z,
                        " pp_loads window=", pp_loads[window_start:(t-1)],
                        " shape_loads=", shape_loads, " rate_loads=", rate_loads);
                }

                pp_states[t] = categorical_rng(softmax(eta));

                int s = pp_states[t];
                real raw_angle;
                pp_speeds[t] = gamma_rng(shape_speeds[s], rate_speeds[s]);
                pp_loads[t] = gamma_rng(shape_loads[s], rate_loads[s]);
                raw_angle = von_mises_rng(loc[s], kappa[s]);
                pp_angles[t] = atan2(sin(raw_angle), cos(raw_angle));
            }
        }
    }
}