import pandas as pd
from pathlib import Path

INPUT_FILE = "generated_data/causation_correlation_pairs.csv"
OUTPUT_DIR = "prepared_data"
OUTPUT_FILE = f"{OUTPUT_DIR}/causation_correlation_prompts.csv"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)
print("Total rows:", len(df))

def build_prompt(text):
    return f"""Consider the causality in the following scenario:

Scenario:
{text}

The amount of causality is
"""
df["prompt_causal"] = df["causation"].apply(build_prompt)
df["prompt_corr"] = df["correlation"].apply(build_prompt)

df_prompts = df[["prompt_causal", "prompt_corr"]]

df_prompts.to_csv(OUTPUT_FILE, index=False)

print("Saved prompts to:", OUTPUT_FILE)
