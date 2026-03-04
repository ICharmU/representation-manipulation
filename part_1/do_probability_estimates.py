import pandas as pd
from dowhy import gcm
import pickle
from generate_synthetic_scm import GenericLinearEquation # needed for the pickled SCM 
import numpy as np

np.random.seed(25)

def do_cf_intervention(scm, interventions, n_samples):
    interventional_data = gcm.interventional_samples(
        scm,
        interventions=interventions,
        num_samples_to_draw=n_samples
    )

    return interventional_data

if __name__ == "__main__":
    with open("generated_data/synthetic/synthetic_0_nodes_4.pkl", "rb") as f:
        scm = pickle.load(f)
    interventions={'X1': lambda x: x*1.1}, # do(X1=5)
    n_samples = 100_000
    interventional_data = do_cf_intervention(scm, interventions, n_samples)

    event_occurrences = (interventional_data['X2'] > 1).sum() # P(X2 > 1 | do(X1 = 1.1*X1))
    total_samples = len(interventional_data)

    probability = event_occurrences / total_samples

    print(f"\nEstimated P(X2 > 1 | do(X1 = 1.1*X1): {probability:.4f}")