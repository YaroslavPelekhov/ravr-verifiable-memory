FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAVR_MEMORY_DB=/data/ravr-memory.sqlite3

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[memory-service]'

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "ravr_science.aml_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
