from __future__ import annotations

import asyncio
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import torch

import torch

from TradingLogic.SignalGenerator.signal_generator import SignalGenerator, TradingSignal

app = FastAPI(title="FinSentBot Signal Generator API")


class MarketData(BaseModel):
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    price: float = Field(..., gt=0.0)
    previous_price: Optional[float] = Field(None, gt=0.0)
    variation: Optional[float] = None
    price_future: Optional[float] = None
    extra: Dict[str, Any] | None = None


class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    market_data: MarketData


class PredictionResponse(BaseModel):
    symbol: str
    action: str
    confidence: float
    price: float
    timestamp: str
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None


def _download_from_s3(uri: str, destination: Path) -> Path:
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 est nécessaire pour télécharger un modèle depuis S3."
        ) from exc

    if not uri.startswith("s3://"):
        raise ValueError("Le paramètre MODEL_S3_URI doit commencer par s3://")

    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError("MODEL_S3_URI invalide. Exemple: s3://bucket/path/model.pth")

    destination.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, key, str(destination))
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover
        raise RuntimeError(f"Impossible de télécharger {uri}: {exc}") from exc

    return destination


@lru_cache()
def _resolve_model_path() -> Path:
    local_path = os.getenv("MODEL_PATH")
    s3_uri = os.getenv("MODEL_S3_URI")

    if local_path:
        resolved = Path(local_path).expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(f"MODEL_PATH introuvable: {resolved}")
        return resolved

    if s3_uri:
        tmp_dir = Path(tempfile.gettempdir()) / "finsentbot_model"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        destination = tmp_dir / "signal_generator.pth"
        return _download_from_s3(s3_uri, destination)

    default_root = Path("models") / "signal_generator"
    if default_root.exists():
        candidates = sorted(
            default_root.glob("*/signal_generator.pth"), key=lambda p: p.stat().st_mtime
        )
        if candidates:
            return candidates[-1].resolve()

    raise RuntimeError(
        "Aucun modèle trouvé. Définissez MODEL_PATH ou MODEL_S3_URI."
    )


@lru_cache()
def _load_generator() -> SignalGenerator:
    confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    model_path = _resolve_model_path()
    return SignalGenerator(str(model_path), confidence_threshold=confidence)


def _prepare_market_payload(data: MarketData) -> Dict[str, Any]:
    payload = {
        "sentiment_score": data.sentiment_score,
        "price": data.price,
        "previous_price": data.previous_price or data.price,
    }

    if data.variation is not None:
        payload["variation"] = data.variation
    elif data.previous_price:
        payload["variation"] = (data.price - data.previous_price) / data.previous_price
    else:
        payload["variation"] = 0.0

    if data.price_future is not None:
        payload["price_future"] = data.price_future
    else:
        payload["price_future"] = data.price * (1 + payload["variation"])

    if data.extra:
        payload.update(data.extra)

    return payload


@app.on_event("startup")
async def startup_event() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_generator)


@app.get("/health", tags=["monitoring"])
def health() -> Dict[str, Any]:
    try:
        generator = _load_generator()
        return {
            "status": "ok",
            "model_type": generator.model_type,
            "feature_count": len(generator.feature_cols),
            "sequence_length": generator.sequence_length,
            "confidence_threshold": generator.confidence_threshold,
            "feature_cols": generator.feature_cols,
        }
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        generator = _load_generator()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    market_payload = _prepare_market_payload(request.market_data)

    try:
        signal: TradingSignal = generator.generate_signal(request.symbol, market_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur modèle: {exc}") from exc

    return PredictionResponse(
        symbol=signal.symbol,
        action=signal.action,
        confidence=signal.confidence,
        price=signal.price,
        timestamp=signal.timestamp,
        reasoning=signal.reasoning,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        position_size=signal.position_size,
    )
