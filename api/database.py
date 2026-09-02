"""
database.py
-----------
Handles all database access. We use SQLite because it needs zero setup
(it's just a file: logistics.db) which is ideal for a capstone demo.
Swap the DATABASE_URL below for a PostgreSQL URL later — SQLAlchemy code
doesn't need to change.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./logistics.db"

# check_same_thread=False is needed only for SQLite + FastAPI's threaded
# request handling — not needed for Postgres/MySQL.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class RiskLog(Base):
    """One row = one logged prediction result. This is what the dashboard
    reads from to build the live table and charts."""
    __tablename__ = "risk_logs"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(String, index=True)
    risk_level = Column(String)
    risk_score = Column(Float)
    source = Column(String, default="api")
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create tables if they don't exist yet. Called once on API startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — opens a DB session per-request and always
    closes it afterward, even if the request raises an error."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
