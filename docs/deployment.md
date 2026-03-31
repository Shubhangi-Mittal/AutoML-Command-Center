# Deployment Guide

This repo is set up for a low-cost student-friendly deployment:

- Frontend on Vercel Hobby
- Backend on Fly.io
- PostgreSQL on Neon or Supabase
- Redis on Upstash

## 1. Create the Hosted Services First

### PostgreSQL

Create a free Postgres database on Neon or Supabase and copy the connection string.

Use the pooled/standard SQLAlchemy-friendly connection string that starts with:

```bash
postgresql://...
```

If your provider gives you a `postgres://` URL, convert it to `postgresql://`.

### Redis

Create a free Redis database on Upstash and copy the Redis URL.

It should look like:

```bash
redis://default:<password>@<host>:<port>
```

## 2. Deploy the Backend to Fly.io

This repo already includes a starter [`fly.toml`](../backend/fly.toml).

From the `backend` directory:

```bash
cd backend
fly auth login
fly launch --copy-config --ha=false
fly volumes create automl_data --size 3
```

Set the backend secrets with your real values:

```bash
fly secrets set \
  DATABASE_URL=postgresql://... \
  REDIS_URL=redis://default:... \
  ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
```

Optional AI provider secrets:

```bash
fly secrets set ANTHROPIC_API_KEY=...
fly secrets set GROQ_API_KEY=...
```

Then deploy:

```bash
fly deploy
```

After deploy, verify:

```bash
curl https://<your-fly-app>.fly.dev/health
curl https://<your-fly-app>.fly.dev/docs
```

## 3. Deploy the Frontend to Vercel

Create a new Vercel project from this GitHub repository with these settings:

- Framework: `Next.js`
- Root Directory: `frontend`
- Build Command: default
- Output Directory: default

Set this environment variable in Vercel:

```bash
NEXT_PUBLIC_API_URL=https://<your-fly-app>.fly.dev
```

Deploy the frontend. Once you have the final Vercel URL, make sure the same exact URL is present in Fly as `ALLOWED_ORIGINS`.

If you later add a custom domain, include both origins as a comma-separated list:

```bash
ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app,https://yourdomain.com
```

## 4. Suggested Production Env Vars

Backend:

```bash
PORT=8000
APP_DATA_DIR=/data/app_data
UPLOAD_DIR=/data/uploads
MODEL_DIR=/data/models
DATABASE_URL=postgresql://...
REDIS_URL=redis://default:...
ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
ANTHROPIC_API_KEY=
GROQ_API_KEY=
```

Frontend:

```bash
NEXT_PUBLIC_API_URL=https://<your-fly-app>.fly.dev
```

## 5. Post-Deploy Checks

- Open the Vercel URL
- Upload a dataset
- Run training
- Open the AI agent
- Deploy a model and test a prediction
- Refresh the page and confirm dataset-specific chat history still works

## 6. Important Security Note

Do not commit real API keys into the repository or `docker-compose.yml`.

If a provider key was ever committed previously, rotate it in the provider dashboard before deploying.
