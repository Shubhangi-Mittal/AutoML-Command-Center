from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def list_training_jobs():
	return {"message": "Training router ready"}
