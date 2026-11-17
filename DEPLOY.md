# Deploying Curio

Curio has two parts:

| Part | Host | What it serves |
|---|---|---|
| API (`server/`) | **Render** free web service (Docker) | FastAPI + the trained model files in `server/artifacts/` |
| Web app (`client/`) | **Vercel** | React/Vite single-page app |

No database and no secrets are needed: the model is pre-trained and ships in the repo (~30 MB).

## 1. Push to GitHub

Create an empty repository named `curio` on GitHub (no README), then:

```bash
git remote add origin https://github.com/<you>/curio.git
git push -u origin main
```

CI (`.github/workflows/ci.yml`) runs the Python tests, builds the client, builds the Docker image and checks the container answers a real recommendation.

## 2. API on Render

1. Render → **New → Blueprint** → pick the `curio` repo. It reads `render.yaml`.
2. When asked for `CLIENT_URL`, enter a placeholder for now (e.g. `http://localhost:5174`); you'll set the real Vercel URL in step 4.
3. Deploy. When it's live, open `https://<service>.onrender.com/api/health` → `{"ok":true}`.

Memory: about 210 MB steady, ~290 MB peak while loading the model — inside the free plan's 512 MB.
Free services sleep after 15 minutes idle; the first request after that takes ~30–60 s while it wakes.

## 3. Web app on Vercel

1. Vercel → **Add New → Project** → import `curio`.
2. **Root Directory:** `client` (framework preset: Vite).
3. **Environment variable:** `VITE_API_URL` = your Render URL, e.g. `https://curio-api.onrender.com` (no trailing slash).
4. Deploy. `client/vercel.json` rewrites every path to `index.html`, so links like `/item/…` and `/lab` work on refresh.

## 4. Connect them

Back on Render → `curio-api` → **Environment** → set `CLIENT_URL` to the Vercel URL (e.g. `https://curio-xyz.vercel.app`). Several origins can be comma-separated. Render redeploys automatically.

Check: open the Vercel site → **For you**. Recommendations should load; if you see "Can't reach the Curio API", wait for the API to wake or re-check `VITE_API_URL` / `CLIENT_URL`.

## Retraining (optional, local only)

```bash
pip install -r requirements-dev.txt
python scripts/prepare_data.py     # needs the two files in data/raw (see README)
python scripts/train.py            # ~7 min, rewrites server/artifacts
python scripts/build_personas.py
```

Commit the new `server/artifacts/` and push; Render redeploys with the new model.
