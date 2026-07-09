FROM python:3.11-slim

WORKDIR /srv

# CPU-only wheels — the default PyPI torch build bundles CUDA (~3GB of
# nvidia-* deps) which is dead weight on Cloud Run (no GPU there).
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# deps in their own layer — cached unless requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY model.py .
COPY app/ app/
COPY models/ models/

RUN adduser --disabled-password appuser
USER appuser

# Cloud Run injects $PORT (default 8080) — never hardcode the port
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
