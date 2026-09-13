import json

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.database import get_db, Scan, User
from app.auth import get_optional_user
from app.schemas import DetectRequest, DetectResponse, EngineResult, SiteCategoryResult
from app.utils.input_router import classify_input, normalize_url
from app.utils.url_scraper import scrape_url, ScrapeError
from app.utils.file_parser import parse_file, UnsupportedFileType
from app.utils.report_exporter import build_scan_report_pdf
from app.utils.highlighter import find_highlights
from app.utils.site_category import classify_site_category
from app.engines.base import EngineOutput
from app.engines.perplexity_burstiness import PerplexityBurstinessEngine
from app.engines.website_fingerprint import WebsiteFingerprintEngine
from app.engines.statistical_linguistic import StatisticalLinguisticEngine
from app.engines.transformer_classifier import TransformerClassifierEngine
from app.engines.github_commit_pattern import GithubCommitPatternEngine, GithubRepoError
from app.scoring import combine_scores

router = APIRouter(prefix="/api", tags=["detect"])

# Registered engines. Adding a new engine later = implement it (see
# engines/base.py), add one line here, done.
ENGINES = [
    PerplexityBurstinessEngine(),
    WebsiteFingerprintEngine(),
    StatisticalLinguisticEngine(),
    TransformerClassifierEngine(),
    GithubCommitPatternEngine(),
]
ENGINE_WEIGHTS = {e.engine_id: e.weight for e in ENGINES}


def _safe_run(engine, **kwargs) -> EngineOutput:
    """One engine failing (e.g. a model download hiccup) shouldn't take
    down the whole scan — return a low-confidence error result instead."""
    try:
        return engine.run(**kwargs)
    except Exception as e:
        return EngineOutput(
            engine_id=engine.engine_id,
            engine_name=engine.engine_name,
            category=engine.category,
            score=0.0,
            confidence="low",
            summary=f"This engine couldn't complete: {e}",
            details={"error": str(e)},
        )


def _save_scan(db: Session, user, input_type, input_summary, overall_score, band, results):
    scan = Scan(
        owner_id=user.id if user else None,
        input_type=input_type,
        input_summary=input_summary[:200],
        overall_score=overall_score,
        band=band,
        engine_results_json=json.dumps([r.as_dict() for r in results]),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


@router.post("/detect", response_model=DetectResponse)
def detect(
    payload: DetectRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    raw = payload.input_value.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Nothing to analyze — input was empty.")

    input_type = classify_input(raw)
    highlights = []
    site_category = None

    if input_type == "github_repo":
        url = normalize_url(raw)
        results = []
        for engine in ENGINES:
            if engine.applies_to("github_repo"):
                try:
                    results.append(engine.run(url=url))
                except GithubRepoError as e:
                    raise HTTPException(status_code=422, detail=str(e))
        input_summary = url

    elif input_type == "url":
        url = normalize_url(raw)
        try:
            scraped = scrape_url(url)
        except ScrapeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        results = []
        for engine in ENGINES:
            if engine.applies_to("url"):
                results.append(
                    _safe_run(
                        engine,
                        text=scraped["text"],
                        html=scraped["html"],
                        url=url,
                        raw_html_before_js=scraped.get("raw_html_before_js", ""),
                    )
                )
        input_summary = scraped["title"] or url
        site_category = SiteCategoryResult(**classify_site_category(scraped["text"]))

    else:
        results = []
        for engine in ENGINES:
            if engine.applies_to("text"):
                results.append(_safe_run(engine, text=raw))
        input_summary = raw[:120]
        highlights = find_highlights(raw)

    if not results:
        raise HTTPException(
            status_code=422,
            detail="No engines were able to run on this input. Try pasting more text (100+ characters).",
        )

    overall_score, band = combine_scores(results, ENGINE_WEIGHTS)

    scan = _save_scan(db, current_user, input_type, input_summary, overall_score, band, results)

    return DetectResponse(
        input_type=input_type,
        input_summary=input_summary,
        overall_score=overall_score,
        band=band,
        engines=[EngineResult(**r.as_dict()) for r in results],
        scan_id=scan.id,
        highlights=highlights,
        site_category=site_category,
    )


@router.post("/detect/file", response_model=DetectResponse)
def detect_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    content = file.file.read()
    try:
        text = parse_file(file.filename, content)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=415, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Couldn't extract any text from this file.")

    results = []
    for engine in ENGINES:
        if engine.applies_to("file"):
            results.append(_safe_run(engine, text=text))

    overall_score, band = combine_scores(results, ENGINE_WEIGHTS)
    scan = _save_scan(db, current_user, "file", file.filename, overall_score, band, results)

    highlights = find_highlights(text)

    return DetectResponse(
        input_type="file",
        input_summary=file.filename,
        overall_score=overall_score,
        band=band,
        engines=[EngineResult(**r.as_dict()) for r in results],
        scan_id=scan.id,
        highlights=highlights,
    )


@router.get("/detect/{scan_id}/report.pdf")
def download_report(scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    pdf_bytes = build_scan_report_pdf(scan)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=scan-{scan_id}-report.pdf"},
    )
