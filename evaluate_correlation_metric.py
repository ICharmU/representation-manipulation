import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs"
LAYERS = [6, 12, 18]

results = []

for layer in LAYERS:
    causal_acts = np.load(f"{OUTPUT_DIR}/causal_acts_layer_{layer}.npy")
    corr_acts = np.load(f"{OUTPUT_DIR}/corr_acts_layer_{layer}.npy")
    v_cause = np.load(f"{OUTPUT_DIR}/v_cause_layer_{layer}.npy")

    causal_scores = causal_acts @ v_cause
    corr_scores = corr_acts @ v_cause

    causal_mean = float(np.mean(causal_scores))
    corr_mean = float(np.mean(corr_scores))
    gap = causal_mean - corr_mean

    results.append({
        "layer": layer,
        "causal_mean_score": causal_mean,
        "correlation_mean_score": corr_mean,
        "score_gap": gap
    })

    print(f"Layer {layer}")
    print(f"  causal mean score      = {causal_mean:.4f}")
    print(f"  correlation mean score = {corr_mean:.4f}")
    print(f"  gap                    = {gap:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/correlation_metric_results.csv", index=False)

print("Saved results to outputs/correlation_metric_results.csv")