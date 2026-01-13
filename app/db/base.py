from typing import Any
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime
from datetime import datetime
from sqlalchemy.ext.declarative import declared_attr

class Base(DeclarativeBase):
    """
    Base class for all ORM models.
    Includes standard audit columns and table naming convention.
    """
    
    @declared_attr
    def __tablename__(cls) -> str:
        # Auto-generate table name from class name (e.g., User -> users)
        return cls.__name__.lower() + "s"

    # Common fields for all tables
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict[str, Any]:
        """Generic dictionary conversion."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}