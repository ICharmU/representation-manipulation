from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sim(scenario, causal_prompt, correlation_prompt):
    # compare embeddings in the context of no scenario and no difference in perspective
    embedding1 = model.encode(causal_prompt) - model.encode("causation") - model.encode(scenario)
    embedding2 = model.encode(correlation_prompt) - model.encode("correlation") - model.encode(scenario)

    similarity_score = util.dot_score(embedding1, embedding2).item() / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    return similarity_score

import pandas as pd
from pathlib import Path

pair_fps = ["causation_correlation_pairs.csv", "mechanism_temporal_pairs.csv"]

for pair_fp in pair_fps:
    df = pd.read_csv(Path("generated_data") / pair_fp)
    df["embedded_similarity"] = df.apply(lambda x: get_sim(x.iloc[0], x.iloc[1], x.iloc[2]), axis=1) # [scenario - p1 - p2] csv structure
    df["valid_embedding"] = df["embedded_similarity"] > 0 # negative embedding indicates that word invariance was significant
    df.to_csv(Path("validated_data") / pair_fp, index=False)

    num_invalid_embeddings = df.shape[0] - df["valid_embedding"].sum()
    if num_invalid_embeddings > 0:
        print(f"There are {num_invalid_embeddings} invalid embeddings in {pair_fp}. It is recommended that these pairs be regenerated")