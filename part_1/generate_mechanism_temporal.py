from google import genai
from pydantic import BaseModel
import pandas as pd

from pathlib import Path
import json, os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

gemini_model = "gemini-3-flash-preview"

prompt = """
Come up with a causal mechanism-temporal sequence pair.
This is a specific case of causal-correlation pairs where the correlation is temporal.
First, produce a causal mechanism scenario.
Next, twist the mechanism scenario to instead be a temporal sequence.
For both scenarios do not make the causal or correlation explicit, but rather implicit by providing select information.
Be a little extra creative so you avoid duplicating scenarios when asking this question again later.
"""

class MechanismTemporalPair(BaseModel):
    scenario: str
    mechanism: str
    temporal: str

scenarios = list()
pairs = list()
n_pairs = 200

for _ in range(n_pairs):
    res = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config = {
            "response_mime_type": "application/json",
            "response_json_schema": MechanismTemporalPair.model_json_schema()
        }
    )

    out = json.loads(res.text)
    pairs.append(out)
    scenarios.append(out["scenario"])

df = pd.DataFrame(pairs)
df.to_csv(Path("generated_data") / "mechanism_temporal_pairs.csv", index=False)