import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.sql import func

from app.database import Base


class Experiment(Base):
	__tablename__ = "experiments"

	id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
	dataset_id = Column(String, ForeignKey("datasets.id"))
	name = Column(String)
	description = Column(String)
	best_job_id = Column(String, ForeignKey("training_jobs.id"))
	optimization_metric = Column(String, default="f1")
	status = Column(String, default="running")
	created_at = Column(DateTime, server_default=func.now())
