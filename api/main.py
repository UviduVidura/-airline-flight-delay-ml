"""FastAPI application entry point."""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Airline Flight Delay Prediction API",
    description="Explainable AI-based airline delay prediction and decision support service.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Return the application health status."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
