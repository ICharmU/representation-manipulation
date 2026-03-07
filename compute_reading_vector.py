from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

OUTPUT_DIR = "outputs"
LAYERS = [6, 12, 18]

for layer in LAYERS:
    causal_file = f"{OUTPUT_DIR}/causal_acts_layer_{layer}.npy"
    corr_file = f"{OUTPUT_DIR}/corr_acts_layer_{layer}.npy"

    causal_acts = np.load(causal_file)
    corr_acts = np.load(corr_file)

    differences = causal_acts - corr_acts

    pca = PCA(n_components=1)
    pca.fit(differences)

    v_cause = pca.components_[0]

    # normalize to unit length
    v_cause = v_cause / np.linalg.norm(v_cause)

    save_file = f"{OUTPUT_DIR}/v_cause_layer_{layer}.npy"
    np.save(save_file, v_cause)

    print(f"Saved {save_file}")