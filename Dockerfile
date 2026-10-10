# Backend API for the deployed website (Render free web service).
# The frontend is deployed separately on Vercel (webapp/frontend). Secondary model = Gemini (API).
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PTP_LLM_PROVIDER=gemini
WORKDIR /app

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# only what the backend needs (see .dockerignore)
COPY ptp_copilot ptp_copilot
COPY webapp/__init__.py webapp/__init__.py
COPY webapp/backend webapp/backend
COPY csv_folder csv_folder
COPY outputs outputs

# Render injects PORT (default 10000)
CMD ["sh", "-c", "python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
