"""Pipeline model for storing reusable chunking workflows."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String, Text

from chonkie.api.database import Base


class Pipeline(Base):
    """A reusable pipeline configuration."""

    __tablename__ = "pipelines"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text)
    config = Column(JSON, nullable=False)  # {"steps": [...]}
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        """Convert to a plain dict for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
