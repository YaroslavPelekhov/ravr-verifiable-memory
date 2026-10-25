"""Deterministic, provenance-preserving memory for external agent evaluations.

The store deliberately does not generate answers.  It keeps immutable source
messages, isolates every evaluation user, and ranks evidence with a transparent
lexical/temporal rule.  This makes it suitable for the Agent Memory Leaderboard
Add/Search boundary, where the participant owns retrieval and the platform owns
answering and grading.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)
_TEMPORAL_LATEST = frozenset(
    {
        "current",
        "currently",
        "latest",
        "last",
        "new",
        "newest",
        "now",
        "recent",
        "recently",
        "today",
    }
)
_TEMPORAL_EARLIEST = frozenset({"earliest", "first", "initial", "initially", "oldest"})
_NEGATIONS = frozenset({"no", "not", "never", "none", "neither", "without", "don't", "didn't"})
_QUESTION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "should",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any, *, length: int | None = None) -> str:
    output = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return output if length is None else output[:length]


def tokenize(text: str, *, remove_stopwords: bool = False) -> tuple[str, ...]:
    tokens = tuple(
        match.group(0).casefold().replace("’", "'") for match in _TOKEN_RE.finditer(text)
    )
    if remove_stopwords:
        return tuple(
            token for token in tokens if token not in _QUESTION_STOPWORDS and len(token) > 1
        )
    return tokens


def _timestamp_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("timestamp must be an integer number of Unix milliseconds")
    try:
        output = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp must be an integer number of Unix milliseconds") from error
    if not -62_135_596_800_000 <= output <= 253_402_300_799_999:
        raise ValueError("timestamp is outside the supported datetime range")
    return output


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    session_id: str
    request_id: str
    ordinal: int
    role: str
    content: str
    timestamp_ms: int | None
    inserted_at_ms: int
    polarity: int

    @property
    def evidence_text(self) -> str:
        labels = [f"role={self.role}"]
        if self.timestamp_ms is not None:
            labels.insert(0, f"timestamp={self.timestamp_ms}")
        return f"[{'; '.join(labels)}]\n{self.content}"


class VerifiableMemoryStore:
    """SQLite-backed Add/Search store with strict user isolation and audit receipts."""

    version = "ravr-verifiable-memory-v1.0.1"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_request (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    inserted_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL REFERENCES add_request(request_id),
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp_ms INTEGER,
                    inserted_at_ms INTEGER NOT NULL,
                    polarity INTEGER NOT NULL,
                    token_json TEXT NOT NULL,
                    UNIQUE(request_id, ordinal)
                );

                CREATE INDEX IF NOT EXISTS memory_user_idx
                    ON memory(user_id, inserted_at_ms, row_id);
                CREATE INDEX IF NOT EXISTS memory_session_idx
                    ON memory(user_id, session_id, row_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    content,
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TABLE IF NOT EXISTS search_receipt (
                    receipt_id TEXT PRIMARY KEY,
                    user_id_hash TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    version TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _validate_identity(value: Any, field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        output = value.strip()
        if not output:
            raise ValueError(f"{field} must be a non-empty string")
        if len(output) > 512:
            raise ValueError(f"{field} is too long")
        return output

    @staticmethod
    def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")
        normalized: list[dict[str, Any]] = []
        if len(messages) > 100:
            raise ValueError("messages must contain at most 100 items")
        for item in messages:
            if not isinstance(item, dict):
                raise TypeError("every message must be an object")
            role = str(item.get("role") or "").strip()
            content = item.get("content")
            if role not in {"user", "assistant"}:
                raise ValueError("every message role must be user or assistant")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("every message requires non-empty role and string content")
            if len(content) > 250_000:
                raise ValueError("message content is too long")
            normalized.append(
                {
                    "role": role[:64],
                    "content": content.strip(),
                    "timestamp": _timestamp_value(item.get("timestamp")),
                }
            )
        return normalized

    def add(
        self,
        *,
        request_id: Any,
        messages: Any,
        user_id: Any,
        session_id: Any,
    ) -> dict[str, Any]:
        request = self._validate_identity(request_id, "request_id")
        user = self._validate_identity(user_id, "user_id")
        session = self._validate_identity(session_id, "session_id")
        normalized = self._normalize_messages(messages)
        payload_hash = _digest(
            {
                "request_id": request,
                "messages": normalized,
                "user_id": user,
                "session_id": session,
            }
        )
        inserted_at_ms = time.time_ns() // 1_000_000

        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_hash FROM add_request WHERE request_id = ?", (request,)
            ).fetchone()
            if existing is not None:
                if str(existing["payload_hash"]) != payload_hash:
                    raise ValueError("request_id was already used with a different payload")
                return {
                    "success": True,
                    "request_id": request,
                    "user_id": user,
                    "session_id": session,
                }

            connection.execute(
                """
                INSERT INTO add_request(request_id, user_id, session_id, payload_hash, inserted_at_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request, user, session, payload_hash, inserted_at_ms),
            )
            for ordinal, message in enumerate(normalized):
                tokens = tokenize(message["content"])
                polarity = -1 if any(token in _NEGATIONS for token in tokens) else 1
                memory_id = "mem_" + _digest(
                    {
                        "request_id": request,
                        "ordinal": ordinal,
                        "role": message["role"],
                        "content": message["content"],
                        "timestamp": message["timestamp"],
                    },
                    length=24,
                )
                cursor = connection.execute(
                    """
                    INSERT INTO memory(
                        memory_id, user_id, session_id, request_id, ordinal, role, content,
                        timestamp_ms, inserted_at_ms, polarity, token_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        user,
                        session,
                        request,
                        ordinal,
                        message["role"],
                        message["content"],
                        message["timestamp"],
                        inserted_at_ms + ordinal,
                        polarity,
                        _canonical_json(tokens),
                    ),
                )
                connection.execute(
                    "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                    (cursor.lastrowid, message["content"]),
                )

        return {
            "success": True,
            "request_id": request,
            "user_id": user,
            "session_id": session,
        }

    @staticmethod
    def _fts_expression(tokens: Iterable[str]) -> str:
        # Tokens come from our word tokenizer, so quoting is sufficient for FTS5.
        unique = tuple(dict.fromkeys(token for token in tokens if token))
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique[:64])

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=str(row["memory_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            request_id=str(row["request_id"]),
            ordinal=int(row["ordinal"]),
            role=str(row["role"]),
            content=str(row["content"]),
            timestamp_ms=None if row["timestamp_ms"] is None else int(row["timestamp_ms"]),
            inserted_at_ms=int(row["inserted_at_ms"]),
            polarity=int(row["polarity"]),
        )

    def _candidate_rows(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        token_groups: Sequence[Sequence[str]],
        limit: int,
    ) -> list[sqlite3.Row]:
        rows: dict[int, sqlite3.Row] = {}
        for tokens in token_groups:
            expression = self._fts_expression(tokens)
            if not expression:
                continue
            for row in connection.execute(
                """
                SELECT m.*
                FROM memory_fts AS f
                JOIN memory AS m ON m.row_id = f.rowid
                WHERE memory_fts MATCH ? AND m.user_id = ?
                ORDER BY bm25(memory_fts), m.inserted_at_ms DESC
                LIMIT ?
                """,
                (expression, user_id, limit),
            ):
                rows[int(row["row_id"])] = row

        # Temporal questions and vocabulary misses still need a safe fallback.
        for row in connection.execute(
            """
            SELECT * FROM memory WHERE user_id = ?
            ORDER BY COALESCE(timestamp_ms, inserted_at_ms) DESC, row_id DESC
            LIMIT ?
            """,
            (user_id, min(limit, 100)),
        ):
            rows[int(row["row_id"])] = row
        return list(rows.values())

    @staticmethod
    def _document_frequency(rows: Sequence[sqlite3.Row]) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for row in rows:
            frequency.update(set(json.loads(str(row["token_json"]))))
        return frequency

    @staticmethod
    def _overlap_score(
        query_tokens: Sequence[str],
        document_tokens: Sequence[str],
        document_frequency: Counter[str],
        document_count: int,
    ) -> float:
        if not query_tokens:
            return 0.0
        counts = Counter(document_tokens)
        unique_query = tuple(dict.fromkeys(query_tokens))
        weights = {
            token: math.log((document_count + 1.0) / (document_frequency[token] + 0.5)) + 1.0
            for token in unique_query
        }
        denominator = sum(weights.values()) or 1.0
        numerator = sum(weights[token] * min(1.0, counts[token]) for token in unique_query)
        return numerator / denominator

    @staticmethod
    def _bigram_score(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
        query_bigrams = set(itertools.pairwise(query_tokens))
        if not query_bigrams:
            return 0.0
        document_bigrams = set(itertools.pairwise(document_tokens))
        return len(query_bigrams & document_bigrams) / len(query_bigrams)

    def search(
        self,
        *,
        query: Any,
        user_id: Any,
        top_k: Any,
        options: Any = None,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string for the textual track")
        if len(query) > 250_000:
            raise ValueError("query is too long")
        user = self._validate_identity(user_id, "user_id")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        requested_top_k = top_k
        if not 1 <= requested_top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if options is None:
            normalized_options: list[str] = []
        elif isinstance(options, list) and all(isinstance(item, str) for item in options):
            if len(options) > 100:
                raise ValueError("options must contain at most 100 items")
            if any(len(item) > 250_000 for item in options):
                raise ValueError("an option is too long")
            normalized_options = [item.strip() for item in options if item.strip()]
        else:
            raise ValueError("options must be an array of strings when provided")

        query_text = query.strip()
        query_tokens = tokenize(query_text, remove_stopwords=True)
        if not query_tokens:
            query_tokens = tokenize(query_text)
        option_tokens = [tokenize(option, remove_stopwords=True) for option in normalized_options]
        groups: list[Sequence[str]] = [query_tokens, *option_tokens]

        with self._lock, self._connect() as connection:
            rows = self._candidate_rows(
                connection,
                user_id=user,
                token_groups=groups,
                limit=max(200, requested_top_k * 6),
            )
            if not rows:
                return {"data": []}
            document_frequency = self._document_frequency(rows)
            document_count = len(rows)
            ordered_times = sorted(
                {int(row["timestamp_ms"] or row["inserted_at_ms"]) for row in rows}
            )
            time_rank = {value: rank for rank, value in enumerate(ordered_times)}
            time_denominator = max(1, len(ordered_times) - 1)
            temporal_latest = bool(set(query_tokens) & _TEMPORAL_LATEST)
            temporal_earliest = bool(set(query_tokens) & _TEMPORAL_EARLIEST)
            query_negative = bool(set(tokenize(query_text)) & _NEGATIONS)

            scored: list[tuple[float, int, MemoryRecord]] = []
            for row in rows:
                record = self._record(row)
                document_tokens = tuple(json.loads(str(row["token_json"])))
                lexical = self._overlap_score(
                    query_tokens, document_tokens, document_frequency, document_count
                )
                phrase = self._bigram_score(query_tokens, document_tokens)
                option = max(
                    (
                        self._overlap_score(
                            tokens, document_tokens, document_frequency, document_count
                        )
                        for tokens in option_tokens
                    ),
                    default=0.0,
                )
                value = int(row["timestamp_ms"] or row["inserted_at_ms"])
                recency = time_rank[value] / time_denominator
                temporal = (
                    recency if temporal_latest else (1.0 - recency if temporal_earliest else 0.0)
                )
                polarity_match = 1.0 if query_negative == (record.polarity < 0) else 0.0
                score = (
                    0.68 * lexical
                    + 0.12 * phrase
                    + 0.12 * option
                    + 0.07 * temporal
                    + 0.01 * polarity_match
                    + 0.000001 * recency
                )
                scored.append((score, value, record))

            scored.sort(key=lambda item: (item[0], item[1], item[2].memory_id), reverse=True)
            selected = scored[:requested_top_k]
            result = [
                {
                    "id": record.memory_id,
                    "content": record.evidence_text,
                    "score": round(score, 8),
                    **(
                        {
                            "created_at": datetime.fromtimestamp(record.timestamp_ms / 1000, tz=UTC)
                            .isoformat()
                            .replace("+00:00", "Z")
                        }
                        if record.timestamp_ms is not None
                        else {}
                    ),
                }
                for score, _, record in selected
            ]
            receipt_payload = {
                "user_id_hash": _digest(user),
                "query_hash": _digest({"query": query_text, "options": normalized_options}),
                "result_ids": [item["id"] for item in result],
                "version": self.version,
            }
            receipt_id = "search_" + _digest(receipt_payload, length=24)
            connection.execute(
                """
                INSERT OR IGNORE INTO search_receipt(
                    receipt_id, user_id_hash, query_hash, result_hash,
                    result_count, created_at_ms, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    receipt_payload["user_id_hash"],
                    receipt_payload["query_hash"],
                    _digest(result),
                    len(result),
                    time.time_ns() // 1_000_000,
                    self.version,
                ),
            )
        return {"data": result}

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS memories, COUNT(DISTINCT user_id) AS users,
                       COUNT(DISTINCT request_id) AS requests
                FROM memory
                """
            ).fetchone()
        return {
            "version": self.version,
            "memories": int(row["memories"]),
            "users": int(row["users"]),
            "requests": int(row["requests"]),
        }
