"""Pass-through gateway: serves the AML Add/Search/Health contract by relaying to the durable backend."""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request, Response

UPSTREAM = os.environ["RAVR_UPSTREAM_URL"].rstrip("/")
FORWARDED = {"authorization", "x-api-key", "content-type"}
ATTEMPTS = 3

app = FastAPI(title="RAVR memory gateway", docs_url=None, redoc_url=None)
client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=45.0))


async def relay(request: Request, path: str) -> Response:
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARDED}
    for attempt in range(ATTEMPTS):
        try:
            upstream = await client.request(
                request.method, f"{UPSTREAM}{path}", content=body, headers=headers
            )
        except httpx.TransportError:
            if attempt + 1 == ATTEMPTS:
                return Response(
                    content=b'{"detail":{"reason":"backend unavailable"}}',
                    status_code=503,
                    media_type="application/json",
                )
            continue
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )
    raise AssertionError("unreachable")


@app.get("/health")
async def health(request: Request) -> Response:
    return await relay(request, "/health")


@app.post("/add")
async def add(request: Request) -> Response:
    return await relay(request, "/add")


@app.post("/search")
async def search(request: Request) -> Response:
    return await relay(request, "/search")
