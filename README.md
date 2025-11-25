# Cognitive Triad AIOps Platform

This repository contains a small AIOps prototype combining a FastAPI backend (RCA + forecasting + LLM helpers), a React frontend, and training scripts for machine learning models.

This README summarizes the project layout, prerequisites, setup and run instructions for local development on Windows (PowerShell), plus pointers for training models and troubleshooting common issues.

---

## Repository layout

- `backend/` — FastAPI server, `server.py`, Python dependencies in `requirements.txt`, `ml_models/` for saved models.
- `frontend/` — React app (CRACO + Tailwind) with `package.json` and build output in `frontend/build/`.
- `ml_training/` — standalone training scripts: `train_rca_models.py`, `train_forecasting_models.py`.
- `tests/` — test helpers and unit tests (if present).
- Misc docs: `AUTH_GUIDE.md`, `ML_TRAINING_GUIDE.md`, `auth_testing.md`, and other notes.

---

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ and `npm` (or `yarn`)
- MongoDB instance (connection string required by backend)
- Optional: GPU for faster LSTM training

On Windows (PowerShell), use the following commands to create a Python virtual environment and activate it:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure you have a `.env` file (placed in `backend/`) with at least the following variables set:

```
MONGO_URL=mongodb://<user>:<pass>@host:port
DB_NAME=your_database_name
```

The backend expects a running MongoDB instance and an env var `MONGO_URL` and `DB_NAME`.

---

## Run the backend (development)

From `backend/` with the virtual environment active:

```powershell
# activate venv if not already active
.\.venv\Scripts\Activate.ps1

# start server (reload enabled)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The API is mounted under `/api` (e.g. `http://localhost:8000/api/rca/predict`).

Authentication: the backend uses session tokens and calls an external auth endpoint in `create_session` — see `backend/server.py` for details. You may need to adapt the auth flow or stub it for local development.

---

## Frontend (development)

From the `frontend/` folder:

```powershell
cd frontend
npm install
npm start
```

Notes:
- If `npm install` fails due to peer dependency conflicts (common with mixed versions like `date-fns` and `react-day-picker`) you can either:
	- Run `npm install --legacy-peer-deps` to bypass strict peer resolution, or
	- Update `frontend/package.json` to align versions (e.g. set `date-fns` to a `^3.x` version required by `react-day-picker`).

If you want, I can update `frontend/package.json` to match `react-day-picker`'s peer requirement.

---

## ML training

The `ml_training/` scripts generate synthetic ticket data and train models used by the backend. Example usage (from repo root):

```powershell
# ensure backend venv is active (see Prerequisites)
cd ml_training
python train_rca_models.py
python train_forecasting_models.py
```

These scripts save models and artifacts into `backend/ml_models/`. If the scripts use absolute paths (e.g. `/app/backend/ml_models`), create `backend/ml_models` and run the scripts from the repository root so the expected folders exist.

Output includes:
- `decision_tree.pkl`, `random_forest.pkl`, encoder pickles
- `ticket_history.csv`, SARIMA and LSTM models and plots

---

## Tests

Run tests (if present) from the repository root:

```powershell
pytest -q
```

---

## Notes, caveats & troubleshooting

- The backend relies on environment variables and a reachable MongoDB. If you get a `KeyError` for `MONGO_URL` or `DB_NAME`, create `backend/.env` or export the variables in your shell before starting the app.
- The frontend may require resolving peer dependency warnings. Use `--legacy-peer-deps` or update versions as mentioned above.
- Training scripts can be CPU intensive — reduce epochs or sample sizes when experimenting locally.
- Some scripts and docker/container-oriented paths use absolute paths (e.g. `/app/backend/...`). Adjust them or run scripts from the expected working directory.

---

## Helpful file pointers

- `backend/server.py` — main FastAPI application and ML integration (RCA, forecasting, auth helpers)
- `backend/requirements.txt` — all Python dependencies for the backend
- `frontend/package.json` — frontend dependencies and scripts
- `ml_training/*` — training scripts and documentation

