# AutoML Command Center

> An end-to-end ML experiment tracking & AutoML platform with an AI agent that orchestrates the entire data science workflow through natural language conversation.

**[Live Demo](https://automl-command-center.vercel.app)** | [API Docs](https://automl-backend.onrender.com/docs) | [Architecture](#architecture)

> **Note:** The backend runs on Render's free tier and may take ~30s to wake up on first visit.

## What It Does

Upload a CSV, and the AI agent will:

1. **Profile** your dataset (distributions, correlations, data quality warnings)
2. **Suggest** a target variable and task type (classification/regression)
3. **Engineer features** automatically (imputation, encoding, scaling, skewness handling)
4. **Train models** in parallel (XGBoost, Random Forest, Logistic/Linear Regression)
5. **Compare results** with detailed metrics and feature importance
6. **Deploy** the best model as a REST API for real-time predictions

All through natural language chat. No clicking through menus.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, SQLAlchemy |
| AI Agent | Claude API (Anthropic), ReAct pattern, 6 custom tools |
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
- ReAct pattern with 6 tools: profile, engineer, train, query, deploy, sample
- Works with Anthropic Claude API (or fallback mode without API key)
- Conversation history and context management

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
