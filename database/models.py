from sqlalchemy import Column, Integer, String, DateTime, JSON, Index, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
import uuid
import os
from typing import List, Optional, Dict, Any

# Base class for all models
Base = declarative_base()

class Report(Base):
    """Database model for storing system health reports."""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String(36), nullable=False, index=True)  # UUID stored as string
    hostname = Column(String(255), nullable=False)
    os = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    checks = Column(JSON, nullable=False)
    
    # Add indexes for common query patterns
    __table_args__ = (
        # Composite index for fast lookups by machine_id and timestamp
        Index('idx_machine_timestamp', 'machine_id', 'timestamp'),
        # Index for timestamp-only queries
        Index('idx_timestamp', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "machine_id": self.machine_id,
            "hostname": self.hostname,
            "os": self.os,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "checks": self.checks
        }

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./reports.db")

# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create a scoped session factory
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database by creating all tables.
    
    Note: In a production environment, use migrations instead.
    """
    Base.metadata.create_all(bind=engine)
