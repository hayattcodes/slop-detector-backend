# Backend — AI Content & Website Detector API

FastAPI backend. Runs entirely on your laptop, no GPU or external services needed for the engines currently included.

## Setup (VS Code terminal)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip3 install -r requirements.txt
cp .env.example .env
```

Open `.env` and set a real `SECRET_KEY` (any random string works for local dev — generate one with
`python3 -c "import secrets; print(secrets.token_hex(32))"`).

## Run it

```bash
uvicorn app.main:app --reload --port 8000
```

- API docs (interactive): http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

The first request to the perplexity/burstiness engine will download the GPT-2 model (~500MB) from
Hugging Face — this only happens once, then it's cached locally.

## What's implemented right now

- **Auth**: signup / login / JWT sessions / per-user scan history — self-hosted, no Firebase
- **`POST /api/detect`**: paste text or a URL — auto-detects which one, scrapes the URL if needed,
  runs whichever engines apply, returns a combined score + per-engine breakdown
- **`POST /api/detect/file`**: upload a `.pdf` / `.docx` / `.txt` file
- **`GET /api/auth/history`**: a logged-in user's past scans

## Engines included so far

| Engine | Runs on | From |
|---|---|---|
| Perplexity & Burstiness | text / file / URL content | Repo 1 + Repo 6 |
| Website Fingerprint | URL only | Your base features + additions |

Next engines to add (see the architecture doc): CNN classifier (Repo 5), statistical/linguistic
heuristics (Repo 4), image detection (Repo 3), the heavy transformer ensemble + LLM-judge
(Repo 4 pattern — needs the Cornell server).

## Database

SQLite, file created automatically at `backend/detector.db` on first run. No setup needed.
