import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUTPUT_DIR = "outputs"
LAYERS = [6, 12, 18]

results = []

for layer in LAYERS:
    causal_file = f"{OUTPUT_DIR}/causal_acts_layer_{layer}.npy"
    corr_file = f"{OUTPUT_DIR}/corr_acts_layer_{layer}.npy"

    causal_acts = np.load(causal_file)
    corr_acts = np.load(corr_file)

    X = np.vstack([causal_acts, corr_acts])
    y = np.array([1] * len(causal_acts) + [0] * len(corr_acts))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        "layer": layer,
        "accuracy": acc
    })

    print(f"Layer {layer}: accuracy = {acc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/probe_results.csv", index=False)

print("Saved probe results to outputs/probe_results.csv")