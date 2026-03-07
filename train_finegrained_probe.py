from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = "part_1/generated_data/mechanism_temporal_pairs.csv"
OUTPUT_DIR = "outputs_finegrained"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_LENGTH = 128
LIMIT_ROWS = 10
LAYERS_TO_SAVE = [6, 12, 18]

Path(OUTPUT_DIR).mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# keep only rows with needed columns
df = df.dropna(subset=["mechanism", "temporal"])

if LIMIT_ROWS is not None:
    df = df.iloc[:LIMIT_ROWS].copy()


def build_prompt(text):
    return f"""Consider the causality in the following scenario:

Scenario:
{text}

The amount of causality is
"""


df["prompt_mechanism"] = df["mechanism"].apply(build_prompt)
df["prompt_temporal"] = df["temporal"].apply(build_prompt)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

hidden_size = model.config.hidden_size
num_hidden_layers = model.config.num_hidden_layers
layers = [l for l in LAYERS_TO_SAVE if l < num_hidden_layers]

mechanism_storage = {
    layer: np.zeros((len(df), hidden_size), dtype=np.float32) for layer in layers
}
temporal_storage = {
    layer: np.zeros((len(df), hidden_size), dtype=np.float32) for layer in layers
}

hook_outputs = {}


def make_hook(layer_idx):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        hook_outputs[layer_idx] = out.detach()
    return hook


hooks = []
for layer in layers:
    h = model.model.layers[layer].register_forward_hook(make_hook(layer))
    hooks.append(h)


def get_last_token_vectors(prompt):
    hook_outputs.clear()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model(**inputs)

    vectors = {}
    for layer in layers:
        hs = hook_outputs[layer][0].float().cpu().numpy()   # [seq_len, hidden_size]
        vectors[layer] = hs[-1]  # last token
    return vectors


for i, row in df.iterrows():
    mechanism_vecs = get_last_token_vectors(row["prompt_mechanism"])
    temporal_vecs = get_last_token_vectors(row["prompt_temporal"])

    for layer in layers:
        mechanism_storage[layer][i] = mechanism_vecs[layer]
        temporal_storage[layer][i] = temporal_vecs[layer]

    print(f"Processed {i + 1}/{len(df)}")

for h in hooks:
    h.remove()

results = []

for layer in layers:
    X = np.vstack([mechanism_storage[layer], temporal_storage[layer]])
    y = np.array([1] * len(mechanism_storage[layer]) + [0] * len(temporal_storage[layer]))

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

    print(f"Layer {layer}: fine-grained accuracy = {acc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/finegrained_probe_results.csv", index=False)

print("Saved fine-grained probe results to outputs_finegrained/finegrained_probe_results.csv")