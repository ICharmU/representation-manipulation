# Representation Manipulation
## Part 1
### Setup:  
Create a new conda environment using the `p1_env.yml` file.
Run all scripts from the `/part_1/` directory while using the `rm_p1` environment.
  
### Usage:  
1. (opt.) If looking for new causation-correlation and mechanism-temporal pairs specify `GENERATE_PAIRS = True` in `runner.py`. **Note**: this takes ~40 minutes each if not running in separate instances since the API calls are made individually on Google's servers when creating the 200 pair set.
2. Run `runner.py` with `GENERATE_SCMS = True` to generate SCMS.
3. Modify and run `do_probability_estimates.py` to estimate the probability of Do(X) corresponding with a change in another causal mechanism Y.
