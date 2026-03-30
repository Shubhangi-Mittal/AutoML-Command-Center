# Deployment Guide

This repo is set up to deploy:

- Frontend on Vercel
- Backend on Fly.io

## 1. Frontend on Vercel

Create a new Vercel project from this GitHub repository.

Use these settings:

- Framework: `Next.js`
- Root Directory: `frontend`
- Build Command: default
- Output Directory: default

Set this environment variable in Vercel:

```bash
NEXT_PUBLIC_API_URL=https://<your-fly-app>.fly.dev
```

After the first backend deploy, add your Vercel domain to the backend CORS setting:

```bash
ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
```

If you later add a custom domain, include both domains in `ALLOWED_ORIGINS` as a comma-separated list.

## 2. Backend on Fly.io

Install and authenticate `flyctl`, then deploy from the `backend` directory.

```bash
cd backend
fly auth login
fly launch --copy-config --ha=false
```

This repo already includes a starter [`fly.toml`](../backend/fly.toml).

Create a Fly volume for persistent app state:

```bash
fly volumes create automl_data --size 3
```

Set the required secrets:

```bash
fly secrets set \
  DATABASE_URL=postgresql://... \
  REDIS_URL=redis://... \
  ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
```

Optional secrets:

```bash
fly secrets set ANTHROPIC_API_KEY=...
fly secrets set GROQ_API_KEY=...
```

Then deploy:

```bash
fly deploy
```

## 3. Backend Dependencies You Need

For Fly.io, this backend expects:

- A PostgreSQL database
- A Redis instance for Celery/background work and training progress events

Good options:

- Fly Postgres or a managed Postgres provider like Neon
- Upstash Redis or another managed Redis provider

## 4. Suggested Production Env Vars

Backend:

```bash
PORT=8000
APP_DATA_DIR=/data/app_data
UPLOAD_DIR=/data/uploads
MODEL_DIR=/data/models
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
ANTHROPIC_API_KEY=
GROQ_API_KEY=
```

Frontend:

```bash
NEXT_PUBLIC_API_URL=https://<your-fly-app>.fly.dev
```

## 5. Post-Deploy Checks

Backend:

```bash
curl https://<your-fly-app>.fly.dev/health
curl https://<your-fly-app>.fly.dev/docs
```

Frontend:

- Open the Vercel URL
- Upload a dataset
- Run training
- Open the AI agent
- Deploy a model and test prediction

## 6. Important Security Note

Do not commit real API keys into the repository or `docker-compose.yml`.

If a real provider key was ever committed previously, rotate it in the provider dashboard before deploying.
