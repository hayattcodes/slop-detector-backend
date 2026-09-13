"""
Pydantic models describing the shape of API requests and responses.
Keeping these separate from the SQLAlchemy models in database.py is
deliberate: schemas.py is "what the API speaks", database.py is "what
gets stored".
"""
import datetime
from typing import Optional, List

from pydantic import BaseModel, EmailStr, Field


# ---------- Auth ----------

class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: EmailStr
    display_name: Optional[str] = None

    class Config:
        from_attributes = True


# ---------- Detection ----------

class DetectRequest(BaseModel):
    # Exactly one of these should be provided; the frontend sends whichever
    # matches what the user gave (URL box, text paste, or a file upload
    # is handled by a separate multipart endpoint).
    input_value: str
    deep_scan: bool = False  # if true, also call the heavy Cornell-hosted engines


class EngineResult(BaseModel):
    engine_id: str
    engine_name: str
    category: str  # "content" | "website"
    score: float  # 0-100, higher = more likely AI
    confidence: str  # "low" | "medium" | "high"
    summary: str
    details: dict = {}


class HighlightSpan(BaseModel):
    start: int
    end: int
    phrase: str
    reason: str


class SiteCategoryResult(BaseModel):
    label: str
    confidence: float
    distribution: dict


class DetectResponse(BaseModel):
    input_type: str  # "url" | "text" | "file" | "github_repo"
    input_summary: str
    overall_score: float
    band: str
    engines: List[EngineResult]
    scan_id: Optional[int] = None
    highlights: List[HighlightSpan] = []
    site_category: Optional[SiteCategoryResult] = None


class ScanHistoryItem(BaseModel):
    id: int
    input_type: str
    input_summary: str
    overall_score: float
    band: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True
