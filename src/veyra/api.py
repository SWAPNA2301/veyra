"""Loopback-only dashboard bridge. Uploaded inputs are processed in memory."""

from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from .api_reports import IMAGES, ReportStore, dashboard, precision_recall, shap_manifest
from .batch_prediction import MAX_REQUEST_BYTES, InvalidBatch, csv_payload, score_rows, validated_rows

ROOT = Path(__file__).resolve().parents[2]
ORIGINS = [f"http://{host}:{port}" for host in ("127.0.0.1", "localhost") for port in (8000, 5173, 4173)]
Nonnegative = Annotated[FiniteFloat, Field(ge=0, strict=True)]
Category = Annotated[str, Field(max_length=128, strict=True)]
SCORING_LOCK = Lock()


class ManualTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    TransactionAmt: Nonnegative
    dist1: Nonnegative | None = None
    dist2: Nonnegative | None = None
    ProductCD: Category | None = None
    card4: Category | None = None
    card6: Category | None = None
    P_emaildomain: Category | None = None
    R_emaildomain: Category | None = None
    M1: Category | None = None
    M2: Category | None = None
    M3: Category | None = None
    M4: Category | None = None
    M5: Category | None = None
    M6: Category | None = None
    M7: Category | None = None
    M8: Category | None = None
    M9: Category | None = None
    DeviceType: Category | None = None
    DeviceInfo: Category | None = None
    id_30: Category | None = None
    id_31: Category | None = None
    id_33: Category | None = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize(cls, value):
        if isinstance(value, str):
            return value.strip() or None
        return value


class TransactionExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: dict[str, Annotated[str, Field(strict=True)]] = Field(min_length=1, max_length=22)
    model_run_id: Annotated[str, Field(strict=True, max_length=128)]
    threshold: Annotated[FiniteFloat, Field(strict=True, ge=0, le=1)]
    score: Annotated[FiniteFloat, Field(strict=True, ge=0, le=1)]


app = FastAPI(title="Veyra local dashboard", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def local_requests(request: Request, call_next):
    if request.method == "POST":
        if request.headers.get("origin") not in (None, *ORIGINS):
            return JSONResponse({"detail": "Only the local dashboard may submit predictions."}, status_code=403)
        batch = request.url.path.rstrip("/") == "/api/batch-predict"
        media = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if batch and media not in {"multipart/form-data", "text/csv"}:
            return JSONResponse({"detail": "Use multipart/form-data with field 'file', or text/csv."}, status_code=415)
        if not batch and media != "application/json":
            return JSONResponse({"detail": "A JSON request is required."}, status_code=415)
        if request.headers.get("content-encoding", "identity").lower() != "identity":
            return JSONResponse({"detail": "Compressed request bodies are not supported."}, status_code=415)
        # Stream-limit before JSON parsing; never persist or log request contents.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > (MAX_REQUEST_BYTES if batch else 16_384):
                return JSONResponse({"detail": "Upload exceeds the 5 MB CSV limit." if batch else "Manual input is too large."}, status_code=413)
            body.extend(chunk)
        request._body = bytes(body)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    return JSONResponse({"detail": "Use a finite nonnegative amount, optional nonnegative distances, and text categories (max 128 characters). Identifiers, labels, and other fields are not accepted."}, status_code=422)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/dashboard")
def report_dashboard():
    return dashboard(ROOT)


@app.get("/api/precision-recall")
def curve():
    try:
        return precision_recall(ROOT)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


@app.get("/api/explanations/{key}")
def explanation(key: str):
    store = ReportStore(ROOT)
    if key not in IMAGES or not shap_manifest(store, store.read("training_metrics.json")):
        raise HTTPException(404, "Saved explanation unavailable.")
    path = store.path(IMAGES[key])
    if not path.is_file():
        raise HTTPException(404, "Saved explanation unavailable.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/predict")
def predict(transaction: ManualTransaction):
    try:
        from .dashboard.predictor import model_spec, score_manual
        with SCORING_LOCK:
            spec = model_spec(ReportStore(ROOT))
            return score_manual(transaction.model_dump(), spec)
    except Exception:
        # Do not expose input values, local paths, or serializer tracebacks.
        raise HTTPException(503, "The saved pipeline cannot score right now. Check matching local model artifacts and installed dependencies; report pages remain available.") from None


def _batch_predict(body: bytes, content_type: str):
    try:
        rows = validated_rows(csv_payload(body, content_type), ManualTransaction)
    except InvalidBatch as exc:
        raise HTTPException(exc.status, str(exc)) from None
    except Exception:
        raise HTTPException(400, "The CSV upload could not be read. Check its encoding and format.") from None
    try:
        from .dashboard.predictor import model_spec
        with SCORING_LOCK:
            return score_rows(rows, model_spec(ReportStore(ROOT)))
    except Exception:
        raise HTTPException(503, "The saved pipeline cannot score this batch right now. Check matching local model artifacts and installed dependencies.") from None


@app.post("/api/batch-predict")
async def batch_predict(request: Request):
    # Do not use UploadFile: its default spool may persist large uploads to disk.
    return await run_in_threadpool(_batch_predict, await request.body(), request.headers.get("content-type", ""))


@app.post("/api/transaction-explanation")
def transaction_explanation(payload: TransactionExplanation):
    from .transaction_explanation import StalePrediction, explain_transaction, validate_values
    try:
        row = validate_values(payload.values, ManualTransaction)
    except InvalidBatch as exc:
        raise HTTPException(exc.status, str(exc)) from None
    except Exception:
        raise HTTPException(422, "Submit one valid transaction using the batch input columns.") from None
    try:
        from .dashboard.predictor import model_spec
        with SCORING_LOCK:
            return explain_transaction(row, model_spec(ReportStore(ROOT)), expected_run=payload.model_run_id,
                                       expected_threshold=payload.threshold, expected_score=payload.score)
    except StalePrediction:
        raise HTTPException(409, "This row or model no longer matches the batch result. Analyze the batch again before investigating.") from None
    except Exception:
        raise HTTPException(503, "The saved model cannot explain this transaction right now. Check matching local artifacts and dependencies, then retry.") from None


# Production build can be served locally by the same process; no workspace files
# or model files are mounted. Vite development uses its /api proxy instead.
if (ROOT / "frontend" / "dist" / "index.html").is_file():
    app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dist", html=True), name="dashboard")
