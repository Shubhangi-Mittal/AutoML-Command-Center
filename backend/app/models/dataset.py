import uuid

from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Dataset(Base):
	__tablename__ = "datasets"

	id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
	name = Column(String, nullable=False)
	file_path = Column(String, nullable=False)
	rows = Column(Integer)
	columns = Column(Integer)
	size_bytes = Column(Integer)
	profile = Column(JSON)
	target_column = Column(String)
	task_type = Column(String)
	created_at = Column(DateTime, server_default=func.now())
