"""API route definitions."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/status")
def route_status() -> dict[str, str]:
    """Return a basic service status payload."""
    return {"status": "ok", "service": "airline-flight-delay-ml"}
