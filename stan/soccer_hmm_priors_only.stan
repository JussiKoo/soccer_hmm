data {
    int<lower=0> T;
    int<lower=1> N;
    int n_covs;
    //matrix[T,nCovs+1] covs;   // still needed since beta's shape depends on nCovs
}

parameters {
    positive_ordered[N] mu_speeds;
    vector<lower=0>[N] sigma_speeds;

    vector<lower=0>[N] mu_loads;
    vector<lower=0>[N] sigma_loads;

    vector[N] xangle;
    vector[N] yangle;

    matrix[N*(N-1), n_covs+1] beta;
}

transformed parameters {
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

model {
    // exact same priors as your real model, likelihood removed
    mu_speeds[1] ~ normal(0.5, 0.5);
    mu_speeds[2] ~ normal(1.25, 0.5);
    mu_speeds[3] ~ normal(2.0, 0.5);
    mu_speeds[4] ~ normal(2.75, 0.5);
    mu_speeds[5] ~ normal(3.5, 0.5);
    sigma_speeds ~ student_t(3, 0, 1);

    mu_loads[1] ~ normal(0.0, 1);
    mu_loads[2] ~ normal(1.0, 1);
    mu_loads[3] ~ normal(3.0, 1);
    mu_loads[4] ~ normal(8.0, 1);
    mu_loads[5] ~ normal(9.0, 1);
    sigma_loads ~ student_t(3, 0, 1);

    xangle[1] ~ normal(1, 0.5);
    xangle[2] ~ normal(0.5, 0.5);
    xangle[3] ~ normal(10, 0.5);
    xangle[4] ~ normal(10, 0.5);
    xangle[5] ~ normal(1, 0.5);
    yangle ~ normal(0, 0.5);

    to_vector(beta) ~ normal(0, 1);
}