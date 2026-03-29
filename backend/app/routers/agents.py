from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.agent import get_agent


router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    dataset_id: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"


@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    agent = get_agent()
    result = await agent.chat(
        user_message=request.message,
        session_id=request.session_id or "default",
        dataset_id=request.dataset_id,
        db=db,
    )
    return result


@router.post("/reset")
def reset_agent(request: ResetRequest):
    agent = get_agent()
    agent.reset(request.session_id or "default")
    return {"status": "reset", "session_id": request.session_id}


@router.get("/status")
def agent_status():
    has_api_key = bool(get_agent() and True)
    from app.config import settings
    return {
        "mode": "claude" if settings.ANTHROPIC_API_KEY else "fallback",
        "api_key_set": bool(settings.ANTHROPIC_API_KEY),
    }
