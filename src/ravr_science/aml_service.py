"""FastAPI wrapper for the Agent Memory Leaderboard Add/Search contract."""

from __future__ import annotations

import hmac
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .memory import VerifiableMemoryStore


def _database_path() -> Path:
    return Path(os.getenv("RAVR_MEMORY_DB", "/data/ravr-memory.sqlite3"))


@lru_cache(maxsize=1)
def store() -> VerifiableMemoryStore:
    return VerifiableMemoryStore(_database_path())


def _authorized(authorization: str | None, x_api_key: str | None) -> bool:
    expected = os.getenv("RAVR_MEMORY_API_KEY", "")
    if not expected:
        return False
    bearer = ""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.casefold() in {"bearer", "token"}:
            bearer = value.strip()
    presented = x_api_key or bearer
    return bool(presented) and hmac.compare_digest(presented, expected)


def create_app():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as error:  # pragma: no cover - exercised by deployment setup
        raise RuntimeError("install ravr-science[memory-service] to run the API") from error

    app = FastAPI(
        title="RAVR Verifiable Memory",
        version=VerifiableMemoryStore.version,
        docs_url=None,
        redoc_url=None,
    )

    def require_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not _authorized(authorization, x_api_key):
            raise HTTPException(status_code=401, detail={"reason": "invalid API key"})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": VerifiableMemoryStore.version}

    @app.post("/add")
    def add(payload: dict[str, Any], _: None = Depends(require_key)) -> dict[str, Any]:
        try:
            return store().add(
                request_id=payload.get("request_id"),
                messages=payload.get("messages"),
                user_id=payload.get("user_id"),
                session_id=payload.get("session_id"),
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail={"reason": str(error)}) from error

    @app.post("/search")
    def search(payload: dict[str, Any], _: None = Depends(require_key)) -> dict[str, Any]:
        try:
            return store().search(
                query=payload.get("query"),
                options=payload.get("options"),
                user_id=payload.get("user_id"),
                top_k=payload.get("top_k"),
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail={"reason": str(error)}) from error

    return app


app = create_app()
