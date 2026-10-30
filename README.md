# RAVR memory gateway

Stateless pass-through for the Agent Memory Leaderboard contract. It relays `GET /health`,
`POST /add` and `POST /search` (with the caller's authentication headers) to the durable
Verifiable Evidence Memory v1.1.0 backend named by `RAVR_UPSTREAM_URL`
(source: branch `release/v1.1.0`, commit e164b70763712f9adb6bc3f5b75964b90b20573a).
It stores nothing, holds no secret, transforms no payload, and retries only transport
failures (Add is idempotent by `request_id`). Authentication is enforced by the backend.
