FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN pip install --no-cache-dir --disable-pip-version-check fastapi==0.162.0 uvicorn==0.54.0 httpx==0.28.1
COPY app.py ./

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
