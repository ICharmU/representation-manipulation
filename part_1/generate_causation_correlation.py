from google import genai
from pydantic import BaseModel
import pandas as pd

from pathlib import Path
import json, os
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


class CausationCorrelationPair(BaseModel):
    scenario: str
    causation: str
    correlation: str

gemini_model = "gemini-3-flash-preview"

scenarios = list()
pairs = list()
n_pairs = 200

prompt = """
Come up with a causation correlation pair.
Start by coming up with a causal scenario.
Next, tweak the scenario by replacing words to only imply correlation.
For both scenarios do not make the causal or correlation explicit, but rather implicit by providing select information.
Be a little extra creative so you avoid duplicating scenarios when asking this question again later.
"""

for _ in range(n_pairs):
    res = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config = {
            "response_mime_type": "application/json",
            "response_json_schema": CausationCorrelationPair.model_json_schema()
        }
    )

    out = json.loads(res.text)
    pairs.append(out)
    scenarios.append(out["scenario"])

df = pd.DataFrame(pairs)
df.to_csv(Path("generated_data") / "causation_correlation_pairs.csv", index=False)