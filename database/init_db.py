"""Database initialization utilities."""

from database.database import Base, engine
from database.models import ModelVersion, Prediction  # noqa: F401


def init_db() -> None:
    """Create all configured database tables."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database tables created successfully.")
