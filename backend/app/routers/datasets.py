import os
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.dataset import Dataset
from app.services.profiler import profile_dataset


router = APIRouter()


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
	)

	db.add(dataset)
	db.commit()
	db.refresh(dataset)

	return {"dataset_id": dataset.id, "profile": profile}
