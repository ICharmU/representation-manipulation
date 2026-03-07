from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = "prepared_data/causation_correlation_prompts.csv"
OUTPUT_DIR = "outputs"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_LENGTH = 128
LIMIT_ROWS = 10
LAYERS_TO_SAVE = [6, 12, 18]

Path(OUTPUT_DIR).mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)
if LIMIT_ROWS is not None:
    df = df.iloc[:LIMIT_ROWS]

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

causal_storage = {
    layer: np.zeros((len(df), hidden_size), dtype=np.float32) for layer in layers
}
corr_storage = {
    layer: np.zeros((len(df), hidden_size), dtype=np.float32) for layer in layers
}

# optional token-wise storage
tokenwise_dir = Path(OUTPUT_DIR) / "tokenwise"
tokenwise_dir.mkdir(exist_ok=True)

hook_outputs = {}

def make_hook(layer_idx):
    def hook(module, inputs, output):
        # output is usually [batch, seq_len, hidden_size]
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

def get_vectors_and_tokenwise(prompt, prefix, row_idx):
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

    last_token_vectors = {}

    for layer in layers:
        hs = hook_outputs[layer]                     # [1, seq_len, hidden_size]
        hs = hs[0].float().cpu().numpy()            # [seq_len, hidden_size]

        # save token-wise activations
        np.save(tokenwise_dir / f"{prefix}_tokenwise_layer_{layer}_row_{row_idx}.npy", hs)

        # save last-token activation in memory for PCA/probe pipeline
        last_token_vectors[layer] = hs[-1]

    return last_token_vectors

for i, row in df.iterrows():
    causal_vecs = get_vectors_and_tokenwise(row["prompt_causal"], "causal", i)
    corr_vecs = get_vectors_and_tokenwise(row["prompt_corr"], "corr", i)

    for layer in layers:
        causal_storage[layer][i] = causal_vecs[layer]
        corr_storage[layer][i] = corr_vecs[layer]

    print(f"Processed {i + 1}/{len(df)}")

for layer in layers:
    np.save(f"{OUTPUT_DIR}/causal_acts_layer_{layer}.npy", causal_storage[layer])
    np.save(f"{OUTPUT_DIR}/corr_acts_layer_{layer}.npy", corr_storage[layer])

for h in hooks:
    h.remove()

print("Done")