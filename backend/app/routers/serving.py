from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def serving_status():
	return {"message": "Serving router ready"}
