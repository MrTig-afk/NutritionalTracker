# Contributing

NutriScan is a solo side project, but issues and small PRs are welcome.

## Before you start

Open an issue first for anything bigger than a typo. It saves both of us the
work of a PR that does not fit.

## Run it locally

Backend (Python 3.11+):

```bash
cd backend
python -m venv ../venv && ../venv/Scripts/activate      # Windows; use source ../venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                                      # fill in the values you have; missing ones degrade gracefully
uvicorn main:app --reload --port 8000
```

Frontend (Node 20+):

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev                                               # http://localhost:5173
```

No backend or accounts at all: `VITE_DEMO=1 npm run dev` runs the app on
synthetic in-memory data.

## Checks

```bash
cd frontend && npm run lint && npm run build
python -m py_compile backend/main.py
```

## Branches and commits

- Work on a branch (`feat/<slug>`, `fix/<slug>`), target `main`.
- One short imperative commit line, no trailers.
- User-visible changes bump `CHANGELOG_VERSION` in `frontend/src/version.js`
  and add matching entries to `frontend/public/changelog.v2.json` in the same
  commit. Backend-only changes do not.
- Never commit secrets. `.env` and `*.local` are ignored; keep it that way.

## Security

See [SECURITY.md](SECURITY.md). Please do not open public issues for
vulnerabilities.
