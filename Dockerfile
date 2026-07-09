FROM python:3.11-slim

WORKDIR /srv

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
