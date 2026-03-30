import os
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.dataset import Dataset
from app.services.profiler import profile_dataset
from app.services.project_metadata import metadata_store


router = APIRouter()


@router.get("/")
def list_datasets(db: Session = Depends(get_db)):
	datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
	return [
		_merge_version_metadata({
			"id": d.id,
			"name": d.name,
			"rows": d.rows,
			"columns": d.columns,
			"size_bytes": d.size_bytes,
			"target_column": d.target_column,
			"task_type": d.task_type,
			"created_at": d.created_at.isoformat() if d.created_at else None,
		}, d.id)
		for d in datasets
	]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
	dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
	if not dataset:
		raise HTTPException(status_code=404, detail="Dataset not found")
	return _merge_version_metadata({
		"id": dataset.id,
		"name": dataset.name,
		"rows": dataset.rows,
		"columns": dataset.columns,
		"size_bytes": dataset.size_bytes,
		"target_column": dataset.target_column,
		"task_type": dataset.task_type,
		"profile": dataset.profile,
		"created_at": dataset.created_at.isoformat() if dataset.created_at else None,
	}, dataset.id)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
	if not file.filename:
		raise HTTPException(status_code=400, detail="Filename is required")

	os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
	safe_filename = os.path.basename(file.filename)
	file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

	with open(file_path, "wb") as saved_file:
		shutil.copyfileobj(file.file, saved_file)

	size_bytes = os.path.getsize(file_path)
	max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
	if size_bytes > max_size_bytes:
		os.remove(file_path)
		raise HTTPException(
			status_code=413,
			detail=f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_MB}MB",
		)

	profile = profile_dataset(file_path)
	dataset = Dataset(
		name=safe_filename,
		file_path=file_path,
		rows=profile.get("row_count"),
		columns=profile.get("column_count"),
		size_bytes=size_bytes,
		profile=profile,
		target_column=profile.get("suggested_target"),
		task_type=profile.get("suggested_task_type"),
	)

	db.add(dataset)
	db.commit()
	db.refresh(dataset)
	version_metadata = metadata_store.register_dataset(dataset.id, dataset.name, dataset.file_path)

	return {
		"id": dataset.id,
		"name": dataset.name,
		"rows": dataset.rows,
		"columns": dataset.columns,
		"size_bytes": dataset.size_bytes,
		"target_column": dataset.target_column,
		"task_type": dataset.task_type,
		"profile": dataset.profile,
		"created_at": dataset.created_at.isoformat() if dataset.created_at else None,
		"version_metadata": version_metadata,
	}


@router.get("/{dataset_id}/versions")
def dataset_versions(dataset_id: str):
	return {"versions": metadata_store.list_dataset_versions(dataset_id)}


def _merge_version_metadata(payload: dict, dataset_id: str):
	payload["version_metadata"] = metadata_store.get_dataset(dataset_id)
	return payload
