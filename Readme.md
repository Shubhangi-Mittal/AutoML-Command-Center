# AutoML Command Center

> An end-to-end ML experiment tracking and AutoML platform with an AI agent that orchestrates the full data science workflow through natural language conversation.

**[Live Demo](https://automl-command-center.vercel.app)** | [API Docs](https://automl-backend.onrender.com/docs) | [Architecture](#architecture)

> **Note:** The backend runs on Render's free tier and may take ~30s to wake up on first visit.

## What It Does

Upload a CSV, and the AI agent will:

1. **Profile** your dataset (distributions, correlations, data quality warnings)
2. **Suggest** a target variable and task type (classification/regression)
3. **Engineer features** automatically (imputation, encoding, scaling, skewness handling)
4. **Train models** across multiple baselines (XGBoost, Random Forest, Logistic/Linear Regression)
5. **Compare results** with detailed metrics and feature importance
6. **Deploy** the best model as a REST API for real-time predictions

All through natural language chat. No clicking through menus.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, SQLAlchemy |
| AI Agent | Groq or Claude, ReAct + fast-path routing, dataset-scoped chat |
| ML Pipeline | Scikit-learn, XGBoost, Pandas, NumPy |
| Database | PostgreSQL (Neon) / SQLite (local) |
| Hosting | Vercel (frontend) + Render (backend) |

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────────┐
│   Next.js App   │────▶│             FastAPI Backend               │
│   (Vercel)      │     │                                          │
│                 │     │  ┌─────────┐  ┌──────────┐  ┌─────────┐ │
│  - Dashboard    │     │  │ Profiler│  │ Trainer  │  │ Serving │ │
│  - Upload       │     │  └────┬────┘  └────┬─────┘  └────┬────┘ │
│  - AI Agent Chat│     │       │            │             │      │
│  - Experiments  │     │  ┌────┴────────────┴─────────────┴────┐ │
│  - Deploy       │     │  │         AI Agent (ReAct Loop)      │ │
│                 │     │  │  Tools: profile, train, deploy...  │ │
│                 │     │  └────────────────┬───────────────────┘ │
└─────────────────┘     │                   │                      │
                        │           ┌───────┴───────┐              │
                        │           │  PostgreSQL   │              │
                        │           │  (Neon.tech)  │              │
                        │           └───────────────┘              │
                        └──────────────────────────────────────────┘
```

## Quick Start (Local)

```bash
# Clone
git clone https://github.com/yourusername/automl-command-center
cd automl-command-center

# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env   # Edit with your ANTHROPIC_API_KEY

# Run backend
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Quick Start (Docker)

```bash
cp .env.example .env  # Add your ANTHROPIC_API_KEY
docker-compose up -d
```

## Features

### Data Profiling
- Automatic column type detection (numeric/categorical)
- Missing value analysis with severity warnings
- Distribution histograms for numeric columns
- Correlation analysis with top pairs
- Target variable suggestion

### Model Training
- **Linear models**: Logistic Regression / Ridge Regression
- **XGBoost**: Gradient boosted trees with auto-tuning
- **Random Forest**: Ensemble tree model
- Automated feature engineering (imputation, encoding, scaling)
- Side-by-side metric comparison (F1, accuracy, precision, recall, RMSE, R², MAE)
- Feature importance ranking

### AI Agent
- Natural language interface for the entire ML workflow
- ReAct pattern plus deterministic fast paths for common requests
- Works with Groq, Anthropic Claude, or local fallback mode
- Dataset-specific chat history and session isolation
- Tools for profiling, training, comparison, deployment, prediction templates, and improvement suggestions

### Model Serving
- One-click deployment from experiment results
- REST API for single and batch predictions
- Swagger/OpenAPI documentation at `/docs`

## Deployment

**Frontend → Vercel**
```bash
cd frontend
vercel --prod
```

**Backend → Render.com**
1. Create a Web Service from the GitHub repo
2. Set root directory to `backend`
3. Set environment variables (DATABASE_URL, ANTHROPIC_API_KEY)
4. Deploy

See the full [Deployment Guide](./docs/deployment.md) for detailed instructions.

## Verification

```bash
# Backend syntax check
python3 -m py_compile backend/app/main.py backend/app/routers/*.py backend/app/services/*.py backend/app/tasks/*.py

# Backend unit tests
PYTHONPATH=backend python3 -m unittest discover -s backend/tests

# Frontend production build
cd frontend && npm run build
```

## Why This Is A Strong Student Project

- It solves a full end-to-end workflow instead of a single isolated feature.
- It combines frontend, backend, ML, background jobs, and AI orchestration in one coherent product.
- It shows product thinking: upload, inspect, train, compare, deploy, and test predictions in one place.
- It is still compact enough to explain clearly in an interview, which makes it strong portfolio material.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/datasets/upload` | Upload CSV dataset |
| GET | `/api/datasets/` | List all datasets |
| POST | `/api/training/launch` | Train models |
| GET | `/api/experiments/{id}` | Get experiment details |
| POST | `/api/agent/chat` | Chat with AI agent |
| POST | `/api/serving/deploy` | Deploy a model |
| POST | `/api/serving/predict` | Make predictions |
| GET | `/docs` | Swagger API documentation |

## License

MIT
