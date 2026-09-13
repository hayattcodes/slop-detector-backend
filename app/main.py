from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import auth as auth_router
from app.routers import detect as detect_router

app = FastAPI(
    title="AI Content & Website Detector API",
    description="Unified detection API: text/file content detection + website AI-builder fingerprinting.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "deep_scan_enabled": settings.deep_scan_enabled}


app.include_router(auth_router.router)
app.include_router(detect_router.router)
