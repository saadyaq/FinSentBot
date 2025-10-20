# Deployment Utilities

This directory provides the blocks needed to operationalise the **SignalGenerator** model and its inference API.

## 1. Export the latest checkpoint

```
python deploy/export_signal_generator.py --output-dir build/model --clean
```

Key files created in `build/model/`:

- `signal_generator.pth` — copied from the most recent run under `models/signal_generator/*/`.
- `model_manifest.json` — metadata with model type, features, sequence length and loss curves.

Optional flags:

- `--checkpoint /path/to/signal_generator.pth` — export a specific run.
- `--s3-uri s3://bucket/prefix/` — push the package directly to S3 (requires `boto3` and AWS credentials).

## 2. FastAPI inference service

The service in `deploy/api/app.py` exposes two endpoints:

- `GET /health` — returns model metadata once loaded.
- `POST /predict` — generates a trading signal.

Environment variables:

- `MODEL_PATH` — local path to `signal_generator.pth`.
- `MODEL_S3_URI` — alternative to `MODEL_PATH`; downloads the checkpoint from S3 at startup.
- `CONFIDENCE_THRESHOLD` — overrides the generator's default threshold (0.7).

### Run locally

```
uvicorn deploy.api.app:app --reload --port 8000
```

or via Docker:

```
docker build -t finsentbot-api -f deploy/api/Dockerfile .
docker run -p 8000:8000 -e MODEL_PATH=/app/model/signal_generator.pth \
  -v $(pwd)/models/signal_generator/mlp_20251020_204524:/app/model finsentbot-api
```

To download from S3 instead, provide `-e MODEL_S3_URI=s3://...`.

### Example request

```
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "symbol": "AAPL",
           "market_data": {
             "sentiment_score": 0.12,
             "price": 182.4,
             "previous_price": 180.9
           }
         }'
```

The response contains action, confidence and trading levels (stop-loss / take-profit).

---

With these two steps you can package a trained checkpoint and run the inference service locally. Next steps are to containerise the Streamlit UI and orchestrate both services on AWS (ECS/Fargate, App Runner, etc.).
