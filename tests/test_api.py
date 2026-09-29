"""Tests for the FastAPI app."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """The health endpoint should respond with a healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
