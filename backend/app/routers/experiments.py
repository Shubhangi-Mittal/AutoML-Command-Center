from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def list_experiments():
	return {"message": "Experiments router ready"}
