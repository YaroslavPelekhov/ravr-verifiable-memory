FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAVR_MEMORY_DB=/data/ravr-memory.sqlite3

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --disable-pip-version-check .

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn ravr_science.aml_service:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
