# Build from the repository root:  docker build -f backend/Dockerfile -t riceleaf-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=5000 MODEL_RUNTIME=litert
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY src/ src/
COPY backend/ backend/
COPY models/rice_leaf_model.tflite models/model_metadata.json models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD gunicorn --chdir backend --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app
