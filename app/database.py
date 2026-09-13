"""
Database setup (SQLite via SQLAlchemy) and the two core tables:
- User: for auth + login
- Scan: one row per analysis a logged-in user has run (their history)
"""
import datetime
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    scans = relationship("Scan", back_populates="owner", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable = anonymous scan
    input_type = Column(String, nullable=False)   # "url" | "text" | "file"
    input_summary = Column(String, nullable=False)  # short preview/label shown in history
    overall_score = Column(Float, nullable=False)
    band = Column(String, nullable=False)  # Clean / Low Risk / Suspicious / Likely AI / High Confidence AI
    engine_results_json = Column(Text, nullable=False)  # full per-engine breakdown, stored as JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="scans")

    def engine_results(self):
        return json.loads(self.engine_results_json)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
