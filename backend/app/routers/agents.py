from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def agent_status():
	return {"message": "Agent router ready"}
