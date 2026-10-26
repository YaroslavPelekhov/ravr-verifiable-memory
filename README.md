# Verifiable Evidence Memory v1.0.2

A deterministic, evidence-only long-term memory service for auditable agent
evaluation. It implements the Agent Memory Leaderboard Cycle 2 Textual Memory
Add/Search contract without an embedding model, LLM, answer generator, or external
knowledge source.

## Method

Add persists every source message unchanged under a strict `user_id` isolation
boundary. Search creates a bounded candidate set using SQLite FTS5 and applies a
transparent ranking rule over lexical overlap, adjacent query terms, answer-option
evidence, temporal intent, polarity, and a deterministic recency tie-break. It returns
ranked source evidence, never a final answer. Version 1.0.2 expands only the first ten
ranked candidates with the two adjacent turns on each side. The bounded expansion
preserves conversational context without increasing the result count.

The service adds four verification properties that are useful beyond raw retrieval:

- idempotent Add keyed by the exact request payload;
- stable content-bound memory identifiers;
- strict cross-user isolation in both candidate generation and fallback retrieval;
- append-only search receipts binding query hashes to ordered result hashes.
- cross-Add context windows ordered within `session_id`, including when one logical
  session is split across multiple 20-message requests.

No third-party memory implementation or benchmark answer data is used. The
implementation is original work by Yaroslav Pelekhov. Runtime dependencies are
FastAPI/Uvicorn and Python's SQLite library.

## Contract

- `GET /health`: unauthenticated 2xx health response.
- `POST /add`: synchronous persistent Add; echoes `request_id`, `user_id`, and
  `session_id` exactly.
- `POST /search`: returns `{"data": [...]}` in relevance order and never exceeds
  `top_k` (1–100).
- Authentication: `X-Api-Key`, `Authorization: Bearer`, or `Authorization: Token`.

Add and Search fail closed when `RAVR_MEMORY_API_KEY` is unset. Health discloses only
service status and version, never stored-content counts.

Only the Textual Memory contract is claimed. Multimodal content arrays are rejected
rather than silently converted.

## Local verification

```bash
python -m pip install -e '.[test]'
pytest -q

RAVR_MEMORY_DB=/tmp/ravr-memory.sqlite3 \
RAVR_MEMORY_API_KEY=local-only \
uvicorn ravr_science.aml_service:app --host 127.0.0.1 --port 8000
```

Smoke requests:

```bash
curl -sS http://127.0.0.1:8000/health

curl -sS -X POST http://127.0.0.1:8000/add \
  -H 'X-Api-Key: local-only' -H 'Content-Type: application/json' \
  -d '{"request_id":"r1","messages":[{"role":"user","timestamp":1704067200000,"content":"Mira moved to Lisbon."}],"user_id":"u1","session_id":"s1"}'

curl -sS -X POST http://127.0.0.1:8000/search \
  -H 'X-Api-Key: local-only' -H 'Content-Type: application/json' \
  -d '{"query":"Where did Mira move?","user_id":"u1","top_k":100}'
```

## Deployment

The included Dockerfile runs one Uvicorn worker so all writes share SQLite's WAL and
the process lock. Mount a persistent volume at `/data` and set:

- `RAVR_MEMORY_DB=/data/ravr-memory.sqlite3`
- `RAVR_MEMORY_API_KEY=<dedicated random evaluation credential>`

Expose HTTPS through the hosting provider. Submit `/add`, `/search`, and `/health` as
separate endpoint URLs. Do not put the credential in Git, a URL, an image, a screenshot,
or an email body.

The expected Cycle 2 scale is small enough for a single persistent SQLite instance.
Advertise conservative concurrency (`Add=1`, `Search=2`) until a hosted load smoke has
confirmed higher values.

### Render fallback

`render.yaml` defines a free Docker web service in Frankfurt with `/health` checks and
prompts for `RAVR_MEMORY_API_KEY` during Blueprint creation. Render's free filesystem is
ephemeral, so this fallback is suitable only for a continuously active evaluation run:
an instance restart, redeploy, or idle spin-down deletes the SQLite database. For a
formal run, warm the service immediately before Add and keep the evaluation continuous.
Use Railway or another provider with a persistent volume when available.

The `deploy/context-v1.0.2` branch is an isolated Smoke deployment wrapper for the
frozen `candidate/context-v1.0.2` implementation. It uses a distinct Render service
name and database path so the approved v1.0.1 endpoint is never replaced in place.

## Evaluation-data lifecycle

Evaluation content is used only to service the active Add/Search run. It is not used
for training, analytics, or dataset reconstruction. Access is limited to the service.
Delete the persistent database within 30 days after the evaluation ends, or earlier if
the organizers request it.

## Version disclosure

- System: Verifiable Evidence Memory
- Version: 1.0.2
- Track: Textual Memory
- Division: Open-source Methods
- Embedding components: none
- LLM components: none
- Reranker: deterministic transparent rule
- External data or APIs: none

The academic model restrictions are therefore not invoked: there is no embedding or
LLM component to substitute. The platform's fixed Answer/Eval stage remains outside
this service.

## License

Apache License 2.0. See `LICENSE`.
