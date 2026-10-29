data {
    int<lower=0> T; // length of the time series
    int<lower=0> N_tracks; // number of tracks
    array[T] int track_ID; // track identifier
    array[T] int segment_ID; // which half of the match
    vector[T] speeds; // step lengths
    vector[T] loads; // player loads
    vector[T] angles; // turning angles
    int<lower=1> N; // number of states
    int nCovs; // number of covariates
    matrix[T,nCovs+1] covs; // covariates

}

parameters {
    positive_ordered[N] mu_speeds; // mean of gamma - ordered
    vector<lower=0>[N] sigma_speeds; // SD of gamma
    
    vector<lower=0>[N] mu_loads;
    vector<lower=0>[N] sigma_loads; // SD of gamma

    // unconstrained angle parameters
    vector[N] xangle;
    vector[N] yangle; 

    // regression coefficients for transition probabilities
    // allow only transitions i -> i, i -> i+1, i -> i-1?
    // beta parameters = 2*(N-1) = 2N - 2 vs original N*(N-1)
    matrix[N*(N-1),nCovs+1] beta;
}

transformed parameters {

    //gamma parameters for speed
    vector<lower=0>[N] shape_speeds;
    vector<lower=0>[N] shape_loads;

    //gamma params for playerload
    vector<lower=0>[N] rate_speeds;
    vector<lower=0>[N] rate_loads;

    //von mises parameters for angles
    vector<lower=-pi(),upper=pi()>[N] loc;
    vector<lower=0>[N] kappa;
    
    for(n in 1:N) {
        // derive turning angle mean and concentration
        loc[n] = atan2(yangle[n], xangle[n]);
        kappa[n] = sqrt(xangle[n]*xangle[n] + yangle[n]*yangle[n]);
    
        // transform mean and SD to shape and rate
        shape_speeds[n] = mu_speeds[n]*mu_speeds[n]/(sigma_speeds[n]*sigma_speeds[n]);
        shape_loads[n] = mu_loads[n]*mu_loads[n]/(sigma_loads[n]*sigma_loads[n]);
    
        rate_speeds[n] = mu_speeds[n]/(sigma_speeds[n]*sigma_speeds[n]);
        rate_loads[n] = mu_loads[n]/(sigma_loads[n]*sigma_loads[n]);
    }
}

model {
    //log likelihood
    vector[N] logp;
    vector[N] logptemp;

    //log transition matrix
    //array[T] matrix[N,N] gamma;
    array[T] matrix[N,N] log_gamma;
    //array[T] matrix[N,N] log_gamma_tr;

    // priors
    mu_speeds[1] ~ normal(0.25, 0.5);
    mu_speeds[2] ~ normal(1.0, 0.5);
    mu_speeds[3] ~ normal(1.75, 0.5);
    mu_speeds[4] ~ normal(2.5, 0.5);
    mu_speeds[5] ~ normal(3.25, 0.5);
    sigma_speeds ~ student_t(3, 0, 1);
    
    mu_loads ~ normal(5, 5);
    sigma_loads ~ student_t(3, 0, 1);

    xangle ~ normal(5, 5);

    //turn angles are mostly near zero
    yangle ~ normal(0, 0.5);

    to_vector(beta) ~ normal(0, 1);

    //derive array of (log-)transition probabilities from betas and covariates
    //multinomial logit-link
    for(t in 1:T) {
        row_vector[N] eta;

        for (i in 1:N) {
            int betarow = (i-1)*(N-1)+1;
            int k = 1;

            for (j in 1:N) {
                if (i == j) {
                    eta[j] = 0;  // reference
                } else {
                    eta[j] = dot_product(beta[betarow], covs[t]);
                    betarow += 1;
                }
            }
            //row of log-transition probs, one row sum equals 1
            log_gamma[t,i] = eta - log_sum_exp(eta);
        }
    }

    //likelihood computation implemented with forward-algorithm
    for (t in 1:T) {
        // initialise forward variable if first obs of track or segment
        if (t==1 || track_ID[t]!=track_ID[t-1] || segment_ID[t]!=segment_ID[t-1]) {
            for (n in 1:N) {
                logp[n] = -log(N);

                logp[n] += gamma_lpdf(speeds[t] | shape_speeds[n], rate_speeds[n]);
                logp[n] += gamma_lpdf(loads[t] | shape_loads[n], rate_loads[n]);
                logp[n] += von_mises_lpdf(angles[t] | loc[n], kappa[n]);
            }
        } else {
            for (n in 1:N) {
                logptemp[n] = log_sum_exp(to_vector(log_gamma[t, :, n]) + logp);
                logptemp[n] = logptemp[n] + gamma_lpdf(speeds[t] | shape_speeds[n], rate_speeds[n]);
                logptemp[n] = logptemp[n] + gamma_lpdf(loads[t] | shape_loads[n], rate_loads[n]);
                logptemp[n] = logptemp[n] + von_mises_lpdf(angles[t] | loc[n], kappa[n]);
            }

            logp = logptemp;
        }

        // add log forward variable to target at the end of each track or half
        if (t==T || track_ID[t+1]!=track_ID[t] || segment_ID[t+1]!=segment_ID[t]) 
            target += log_sum_exp(logp);
            
    }
}

generated quantities {

    //Viterbi

    array[T] int<lower=1,upper=N> viterbi;

    {
        row_vector[N] eta;

        array[T] matrix[N,N] log_gamma;
        
        for(t in 1:T) {
            for (i in 1:N) {
                int betarow = (i-1)*(N-1)+1;
                int k = 1;

                for (j in 1:N) {
                    if (i == j) {
                        eta[j] = 0;  // reference
                    } else {
                        eta[j] = dot_product(beta[betarow], covs[t]);
                        betarow += 1;
                    }
                }
                log_gamma[t,i] = eta - log_sum_exp(eta);
            }
        }

        //for(t in 1:T)
        //    for(i in 1:N)
        //        for(j in 1:N)
        //            log_gamma_tr[t,j,i] = log_gamma[t,i,j];

        // Viterbi algorithm (most likely state sequence)
        real max_logp;
        array[T, N] int back_ptr; //which state n at t-1 led to best path in t
        array[T, N] real best_logp; //log probability of the most likely path ending in state n at time t
        
        for (t in 1:T) {
            if (t==1 || track_ID[t]!=track_ID[t-1] || segment_ID[t]!=segment_ID[t-1]) {
                for(n in 1:N) {
                    real logp = 0;
                    if (speeds[t] > 0)
                        logp = logp + gamma_lpdf(speeds[t] | shape_speeds[n], rate_speeds[n]);
                    if (loads[t] > 0)
                        logp = logp + gamma_lpdf(loads[t] | shape_loads[n], rate_loads[n]);
                    if (angles[t] > (-pi()))
                        logp = logp + von_mises_lpdf(angles[t] | loc[n], kappa[n]);
                    back_ptr[t, n] = 1;
                    best_logp[t, n] = logp;
                }
            } else {
                for (n in 1:N) {
                    best_logp[t, n] = negative_infinity();
                    for (j in 1:N) {
                        real logp = 0;
                        logp = best_logp[t-1, j] + log_gamma[t,j,n]; //transition from state j to n
                        if (speeds[t] > 0)
                            logp = logp + gamma_lpdf(speeds[t] | shape_speeds[n], rate_speeds[n]);
                        if (loads[t] > 0)
                            logp = logp + gamma_lpdf(loads[t] | shape_loads[n], rate_loads[n]);
                        if (angles[t] > (-pi())) 
                            logp = logp + von_mises_lpdf(angles[t] | loc[n], kappa[n]);                     
                        if (logp > best_logp[t, n]) {
                            back_ptr[t, n] = j;
                            best_logp[t, n] = logp;
                        }
                    }
                }
            }
        }

        for(t0 in 1:T) {
            int t = T - t0 + 1;
            if (t==T || track_ID[t+1]!=track_ID[t] || segment_ID[t+1]!=segment_ID[t]) {
                max_logp = best_logp[t, 1];
                int argmax = 1;
                for (n in 2:N)
                    if (best_logp[t, n] > max_logp) {
                        max_logp = best_logp[t, n];
                        argmax = n;
                    }
                viterbi[t] = argmax;
            } else {
                viterbi[t] = back_ptr[t+1, viterbi[t+1]];
            }
        }
    }

}