"""Write the demo personas' real rating histories next to the model files.

Persona user IDs come from server/curio/personas.json (anonymised IDs from the public dataset).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ART = ROOT / "server" / "artifacts"

personas = json.loads((ROOT / "server" / "curio" / "personas.json").read_text(encoding="utf-8"))
df = pd.concat([pd.read_parquet(DATA / f"{s}.parquet") for s in ("train", "valid", "test")])
out = {}
for p in personas:
    if not p["user_id"]:
        out[p["id"]] = []
        continue
    h = df[df.user_id == p["user_id"]].sort_values("timestamp")
    out[p["id"]] = [{"item_id": r.item_id, "rating": float(r.rating)} for r in h.itertuples()]
    print(p["id"], len(out[p["id"]]), "ratings")
(ART / "personas_history.json").write_text(json.dumps(out, indent=2))
