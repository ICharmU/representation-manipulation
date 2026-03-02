import pandas as pd
from dowhy import gcm
import pickle
from generate_synthetic_scm import GenericLinearEquation # needed for the pickled SCM 
import numpy as np

np.random.seed(25)

with open("generated_data/synthetic/synthetic_0_nodes_4.pkl", "rb") as f:
    scm = pickle.load(f)

interventional_data = gcm.interventional_samples(
    scm,
    interventions={'X1': lambda x: x*1.1}, # do(X1=5)
    num_samples_to_draw=10000
)

event_occurrences = (interventional_data['X2'] > 1).sum() # P(X2 > 1 | do(X1 = 1.1*X1))
total_samples = len(interventional_data)

probability = event_occurrences / total_samples

print(f"\nEstimated P(X2 > 1 | do(X1 = 1.1*X1): {probability:.4f}")