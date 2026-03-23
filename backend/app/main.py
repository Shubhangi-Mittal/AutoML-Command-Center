import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.models import dataset, experiment, job  # noqa: F401
from app.routers import agents, datasets, experiments, serving, training


Base.metadata.create_all(bind=engine)
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

app = FastAPI(title="AutoML Command Center", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000", "https://your-app.vercel.app"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(agents.router, prefix="/api/agent", tags=["Agent"])
app.include_router(serving.router, prefix="/api/serving", tags=["Serving"])


@app.get("/health")
def health():
	return {"status": "ok"}
