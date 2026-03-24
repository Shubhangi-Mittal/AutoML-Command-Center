import uuid

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, String
from sqlalchemy.sql import func

from app.database import Base


class TrainingJob(Base):
	__tablename__ = "training_jobs"

	id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
	dataset_id = Column(String, ForeignKey("datasets.id"))
	experiment_id = Column(String, ForeignKey("experiments.id"))
	status = Column(String, default="pending")
	model_type = Column(String)
	hyperparameters = Column(JSON)
	metrics = Column(JSON)
	feature_importance = Column(JSON)
	mlflow_run_id = Column(String)
	training_duration_seconds = Column(Float)
	celery_task_id = Column(String)
	created_at = Column(DateTime, server_default=func.now())
	completed_at = Column(DateTime)
