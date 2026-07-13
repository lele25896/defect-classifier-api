FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard.py .

RUN adduser --disabled-password appuser
USER appuser

# Cloud Run injects $PORT (default 8080) — never hardcode the port
CMD exec streamlit run dashboard.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
