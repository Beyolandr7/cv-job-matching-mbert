import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from db.session import Base


class MatchingHistory(Base):
    """SQLAlchemy model representing the matching_histories table."""
    __tablename__ = "matching_histories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    cv_filename = Column(String(255), nullable=False)
    cv_text = Column(Text, nullable=False)
    job_title = Column(String(255), nullable=False)
    job_description = Column(Text, nullable=False)

    # JSONB for extracted NER entities (skills, education, experience)
    extracted_entities = Column(JSONB, nullable=False, default=dict)

    # Decision & metric fields
    similarity_score = Column(Numeric(5, 4), nullable=False)
    status_category = Column(String(20), nullable=False)  # 'Fit', 'Close Fit', 'Not Fit'

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    # Relationship back to User model
    user = relationship("User", back_populates="histories")
