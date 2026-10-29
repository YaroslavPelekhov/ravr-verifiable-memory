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
_RECOMMEND_CUES = frozenset(
    {
        "suggest",
        "recommend",
        "recommendation",
        "recommendations",
        "idea",
        "ideas",
        "advice",
        "tips",
        "options",
        "should",
        "try",
        "plan",
    }
)
_PREFERENCE_PATTERN = re.compile(
    r"\b(?:i (?:really |absolutely |also |just |truly )?(?:like|love|enjoy|prefer|hate|dislike|"
    r"adore|use|subscribe|listen|watch|play|follow|collect|avoid|am (?:a fan|into|allergic|"
    r"interested|obsessed))|my favou?rite|i always|i usually|i never|i'm (?:a fan|into|allergic|"
    r"interested|obsessed))\b",
    re.IGNORECASE,
)
_ASSISTANT_INTENT = re.compile(
    r"\b(?:you (?:said|told|mentioned|recommended|suggested|provided|gave|listed|wrote|"
    r"explained|described|named|shared|offered)|remind me|previous (?:conversation|chat)|"
    r"our (?:last|previous|earlier) (?:conversation|chat|discussion))\b",
    re.IGNORECASE,
)


def _scope_token(user_id: str) -> str:
    return "u" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]


def _stem(token: str) -> str:
    """Small deterministic suffix stripper; index and query use the same function."""
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2] and token[-1] not in "lsz":
            token = token[:-1]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
        if len(token) > 2 and token[-1] == token[-2] and token[-1] not in "lsz":
            token = token[:-1]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


@dataclass(frozen=True)
class OperatorConfig:
    """Optional retrieval operators; every default leaves the v1.0.2 behaviour unchanged."""

    stem: bool = False
    corpus_idf: bool = False
    session_prior: float = 0.0
    session_diversity: float = 0.0
    role_assistant: float = 0.0
    role_user: float = 0.0
    pref_cue: float = 0.0
    supersession: float = 0.0
    prf_terms: int = 0
    prf_docs: int = 3
    prf_weight: float = 0.0
    iso_dates: bool = False
    passthrough_chars: int = 0
    scoped_fts: bool = False


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


EMPTY_MESSAGE_PLACEHOLDER = "(empty message)"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any, *, length: int | None = None) -> str:
    output = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return output if length is None else output[:length]


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = False,
    normalize_english_possessives: bool = False,
) -> tuple[str, ...]:
    raw_tokens = tuple(
        match.group(0).casefold().replace("’", "'") for match in _TOKEN_RE.finditer(text)
    )
    tokens = tuple(
        token[:-2]
        if normalize_english_possessives and token.endswith("'s") and len(token) > 2
        else token
        for token in raw_tokens
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
    row_id: int
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
        return self.render(False)

    def render(self, iso_dates: bool) -> str:
        labels = [f"role={self.role}"]
        if self.timestamp_ms is not None:
            if iso_dates:
                moment = datetime.fromtimestamp(self.timestamp_ms / 1000, tz=UTC)
                labels.insert(0, f"date={moment.strftime('%Y-%m-%d %a %H:%M')} UTC")
            labels.insert(0, f"timestamp={self.timestamp_ms}")
        return f"[{'; '.join(labels)}]\n{self.content}"


@dataclass(frozen=True)
class RankingWeights:
    """Transparent coefficients for an offline, frozen ranking ablation."""

    lexical: float = 0.68
    bm25: float = 0.0
    phrase: float = 0.12
    option: float = 0.12
    temporal: float = 0.07
    polarity: float = 0.01
    recency_tiebreak: float = 0.000001


class VerifiableMemoryStore:
    """SQLite-backed Add/Search store with strict user isolation and audit receipts."""

    version = "ravr-verifiable-memory-v1.1.0"

    def __init__(
        self,
        path: str | Path,
        *,
        ranking_weights: RankingWeights | None = None,
        normalize_english_possessives: bool = False,
        query_conditioned_temporal: bool = False,
        context_window_radius: int = 0,
        deduplicate_context_windows: bool = False,
        context_window_top_n: int | None = None,
        operators: OperatorConfig | dict[str, Any] | None = None,
    ) -> None:
        self.operators = (
            operators
            if isinstance(operators, OperatorConfig)
            else OperatorConfig(**(operators or {}))
        )
        self.path = Path(path).expanduser().resolve()
        self.ranking_weights = ranking_weights or RankingWeights()
        self.normalize_english_possessives = normalize_english_possessives
        self.query_conditioned_temporal = query_conditioned_temporal
        if not 0 <= context_window_radius <= 5:
            raise ValueError("context_window_radius must be between 0 and 5")
        self.context_window_radius = context_window_radius
        self.deduplicate_context_windows = deduplicate_context_windows
        if context_window_top_n is not None and not 1 <= context_window_top_n <= 100:
            raise ValueError("context_window_top_n must be between 1 and 100")
        self.context_window_top_n = context_window_top_n
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
        tokenizer = (
            "porter unicode61 remove_diacritics 2"
            if self.operators.stem
            else "unicode61 remove_diacritics 2"
        )
        columns = "content, scope" if self.operators.scoped_fts else "content"
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
                    {FTS_COLUMNS},
                    tokenize='{TOKENIZER}'
                );

                CREATE TABLE IF NOT EXISTS term_df (
                    user_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    df INTEGER NOT NULL,
                    PRIMARY KEY (user_id, token)
                ) WITHOUT ROWID;

                CREATE TABLE IF NOT EXISTS search_receipt (
                    receipt_id TEXT PRIMARY KEY,
                    user_id_hash TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    version TEXT NOT NULL
                );
                """.replace("{TOKENIZER}", tokenizer).replace("{FTS_COLUMNS}", columns)
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
            if not isinstance(content, str):
                raise TypeError("every message requires a string content")
            if not content.strip():
                # Public benchmark transcripts contain empty turns (12 in LongMemEval-S); keep the
                # turn so source order and adjacency survive instead of failing the whole Add.
                content = EMPTY_MESSAGE_PLACEHOLDER
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
                tokens = tokenize(
                    message["content"],
                    normalize_english_possessives=self.normalize_english_possessives,
                )
                polarity = -1 if any(token in _NEGATIONS for token in tokens) else 1
                if self.operators.stem:
                    tokens = tuple(_stem(token) for token in tokens)
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
                if self.operators.scoped_fts:
                    connection.execute(
                        "INSERT INTO memory_fts(rowid, content, scope) VALUES (?, ?, ?)",
                        (cursor.lastrowid, message["content"], _scope_token(user)),
                    )
                else:
                    connection.execute(
                        "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                        (cursor.lastrowid, message["content"]),
                    )
                connection.executemany(
                    """
                    INSERT INTO term_df(user_id, token, df) VALUES (?, ?, 1)
                    ON CONFLICT(user_id, token) DO UPDATE SET df = df + 1
                    """,
                    [(user, token) for token in set(tokens)],
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
            row_id=int(row["row_id"]),
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
            if self.operators.scoped_fts:
                match = f'scope:"{_scope_token(user_id)}" AND content:({expression})'
                order = "bm25(memory_fts, 1.0, 0.0)"
            else:
                match = expression
                order = "bm25(memory_fts)"
            for row in connection.execute(
                f"""
                SELECT m.*
                FROM memory_fts AS f
                JOIN memory AS m ON m.row_id = f.rowid
                WHERE memory_fts MATCH ? AND m.user_id = ?
                ORDER BY {order}, m.inserted_at_ms DESC
                LIMIT ?
                """,
                (match, user_id, limit),
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

    @staticmethod
    def _bm25_score(
        query_tokens: Sequence[str],
        document_tokens: Sequence[str],
        document_frequency: Counter[str],
        document_count: int,
        average_document_length: float,
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        counts = Counter(document_tokens)
        k1 = 1.2
        b = 0.75
        length_ratio = len(document_tokens) / max(1.0, average_document_length)
        score = 0.0
        for token in dict.fromkeys(query_tokens):
            frequency = counts[token]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            saturation = frequency * (k1 + 1.0) / (
                frequency + k1 * (1.0 - b + b * length_ratio)
            )
            score += inverse_document_frequency * saturation
        return score

    @staticmethod
    def _user_df(connection: sqlite3.Connection, user_id: str, tokens: Iterable[str]) -> Counter[str]:
        unique = tuple(dict.fromkeys(tokens))
        frequency: Counter[str] = Counter()
        for start in range(0, len(unique), 500):
            batch = unique[start : start + 500]
            marks = ",".join("?" for _ in batch)
            for row in connection.execute(
                f"SELECT token, df FROM term_df WHERE user_id = ? AND token IN ({marks})",
                (user_id, *batch),
            ):
                frequency[str(row["token"])] = int(row["df"])
        return frequency

    def _passthrough(
        self, connection: sqlite3.Connection, user_id: str, top_k: int
    ) -> list[dict[str, Any]] | None:
        """Return the whole memory in order when it is small enough to hand over unranked."""
        limit = self.operators.passthrough_chars
        count, chars = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0) FROM memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not count or count > top_k or chars > limit:
            return None
        result = []
        for row in connection.execute(
            """
            SELECT * FROM memory WHERE user_id = ?
            ORDER BY COALESCE(timestamp_ms, inserted_at_ms), row_id
            """,
            (user_id,),
        ):
            record = self._record(row)
            item: dict[str, Any] = {
                "id": record.memory_id,
                "content": record.render(self.operators.iso_dates),
                "score": 1.0,
            }
            if record.timestamp_ms is not None:
                item["created_at"] = (
                    datetime.fromtimestamp(record.timestamp_ms / 1000, tz=UTC)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            result.append(item)
        return result

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

        cfg = self.operators
        query_text = query.strip()
        raw_query_tokens = tokenize(
            query_text,
            remove_stopwords=True,
            normalize_english_possessives=self.normalize_english_possessives,
        )
        if not raw_query_tokens:
            raw_query_tokens = tokenize(
                query_text,
                normalize_english_possessives=self.normalize_english_possessives,
            )
        raw_option_tokens = [
            tokenize(
                option,
                remove_stopwords=True,
                normalize_english_possessives=self.normalize_english_possessives,
            )
            for option in normalized_options
        ]

        def stemmed(tokens: Sequence[str]) -> tuple[str, ...]:
            return tuple(_stem(token) for token in tokens) if cfg.stem else tuple(tokens)

        query_tokens = stemmed(raw_query_tokens)
        option_tokens = [stemmed(tokens) for tokens in raw_option_tokens]
        groups: list[Sequence[str]] = [raw_query_tokens, *raw_option_tokens]
        all_query_tokens = tokenize(
            query_text, normalize_english_possessives=self.normalize_english_possessives
        )
        recommend_intent = bool(set(all_query_tokens) & _RECOMMEND_CUES)
        assistant_intent = bool(_ASSISTANT_INTENT.search(query_text))

        with self._lock, self._connect() as connection:
            if cfg.passthrough_chars:
                whole = self._passthrough(connection, user, requested_top_k)
                if whole is not None:
                    return {"data": whole}
            rows = self._candidate_rows(
                connection,
                user_id=user,
                token_groups=groups,
                limit=max(200, requested_top_k * 6),
            )
            if not rows:
                return {"data": []}
            document_tokens_by_row = {
                int(row["row_id"]): tuple(json.loads(str(row["token_json"]))) for row in rows
            }
            if cfg.corpus_idf:
                document_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM memory WHERE user_id = ?", (user,)
                    ).fetchone()[0]
                )
                needed = set(query_tokens).union(*option_tokens)
                document_frequency = self._user_df(connection, user, needed)
            else:
                document_frequency = self._document_frequency(rows)
                document_count = len(rows)
            if self.ranking_weights.bm25:
                average_document_length = sum(
                    len(tokens) for tokens in document_tokens_by_row.values()
                ) / max(1, document_count if not cfg.corpus_idf else len(rows))
                raw_bm25 = {
                    row_id: self._bm25_score(
                        query_tokens,
                        tokens,
                        document_frequency,
                        document_count,
                        average_document_length,
                    )
                    for row_id, tokens in document_tokens_by_row.items()
                }
                maximum_bm25 = max(raw_bm25.values(), default=0.0) or 1.0
            else:
                raw_bm25 = {row_id: 0.0 for row_id in document_tokens_by_row}
                maximum_bm25 = 1.0
            temporal_latest = bool(set(raw_query_tokens) & _TEMPORAL_LATEST)
            temporal_earliest = bool(set(raw_query_tokens) & _TEMPORAL_EARLIEST)
            query_negative = bool(set(all_query_tokens) & _NEGATIONS)

            features: list[tuple[MemoryRecord, float, float, float, float, float, int]] = []
            for row in rows:
                record = self._record(row)
                row_id = int(row["row_id"])
                document_tokens = document_tokens_by_row[row_id]
                lexical = self._overlap_score(
                    query_tokens, document_tokens, document_frequency, document_count
                )
                bm25 = raw_bm25[row_id] / maximum_bm25
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
                value = int(
                    row["timestamp_ms"]
                    if row["timestamp_ms"] is not None
                    else row["inserted_at_ms"]
                )
                polarity_match = 1.0 if query_negative == (record.polarity < 0) else 0.0
                features.append((record, lexical, bm25, phrase, option, polarity_match, value))

            global_times = sorted({item[-1] for item in features})
            global_rank = {value: rank for rank, value in enumerate(global_times)}
            global_denominator = max(1, len(global_times) - 1)
            temporal_intent = temporal_latest or temporal_earliest
            if self.query_conditioned_temporal and temporal_intent:
                topical_times = sorted(
                    {
                        value
                        for _, lexical, bm25, phrase, _, _, value in features
                        if lexical > 0 or bm25 > 0 or phrase > 0
                    }
                )
                if not topical_times:
                    topical_times = global_times
            else:
                topical_times = global_times
            topical_rank = {value: rank for rank, value in enumerate(topical_times)}
            topical_denominator = max(1, len(topical_times) - 1)

            maximum_lexical = max((item[1] for item in features), default=0.0) or 1.0
            supersession_rank: dict[int, int] = {}
            supersession_denominator = 1
            if cfg.supersession:
                times = sorted(
                    {item[-1] for item in features if item[1] >= 0.5 * maximum_lexical}
                )
                supersession_rank = {value: rank for rank, value in enumerate(times)}
                supersession_denominator = max(1, len(times) - 1)
            session_best: dict[str, float] = {}
            if cfg.session_prior:
                for record, lexical, *_ in features:
                    best = session_best.get(record.session_id, 0.0)
                    session_best[record.session_id] = max(best, lexical / maximum_lexical)

            def operator_bonus(record: MemoryRecord, lexical: float, value: int) -> float:
                bonus = 0.0
                if cfg.role_assistant and assistant_intent and record.role == "assistant":
                    bonus += cfg.role_assistant
                if cfg.role_user and not assistant_intent and record.role == "user":
                    bonus += cfg.role_user
                if (
                    cfg.pref_cue
                    and recommend_intent
                    and record.role == "user"
                    and _PREFERENCE_PATTERN.search(record.content)
                ):
                    bonus += cfg.pref_cue
                if cfg.supersession and lexical >= 0.5 * maximum_lexical:
                    recency = supersession_rank[value] / supersession_denominator
                    bonus += cfg.supersession * (1.0 - recency if temporal_earliest else recency)
                if cfg.session_prior:
                    bonus += cfg.session_prior * session_best.get(record.session_id, 0.0)
                return bonus

            scored: list[tuple[float, int, MemoryRecord]] = []
            for record, lexical, bm25, phrase, option, polarity_match, value in features:
                recency = global_rank[value] / global_denominator
                topical_recency = (
                    topical_rank[value] / topical_denominator if value in topical_rank else 0.0
                )
                temporal = (
                    topical_recency
                    if temporal_latest
                    else (1.0 - topical_recency if temporal_earliest else 0.0)
                )
                weights = self.ranking_weights
                score = (
                    weights.lexical * lexical
                    + weights.bm25 * bm25
                    + weights.phrase * phrase
                    + weights.option * option
                    + weights.temporal * temporal
                    + weights.polarity * polarity_match
                    + weights.recency_tiebreak * recency
                    + operator_bonus(record, lexical, value)
                )
                scored.append((score, value, record))

            scored.sort(key=lambda item: (item[0], item[1], item[2].memory_id), reverse=True)

            if cfg.prf_terms and cfg.prf_weight and cfg.prf_docs:
                top_docs = scored[: cfg.prf_docs]
                weight_by_token: Counter[str] = Counter()
                feedback_df = self._user_df(
                    connection,
                    user,
                    {t for _, _, r in top_docs for t in document_tokens_by_row[r.row_id]},
                ) if cfg.corpus_idf else document_frequency
                for _, _, record in top_docs:
                    for token, count in Counter(document_tokens_by_row[record.row_id]).items():
                        if token in query_tokens or token in _QUESTION_STOPWORDS or len(token) < 3:
                            continue
                        idf = math.log((document_count + 1.0) / (feedback_df[token] + 0.5)) + 1.0
                        weight_by_token[token] += count * idf
                feedback = tuple(token for token, _ in weight_by_token.most_common(cfg.prf_terms))
                if feedback:
                    boosted: list[tuple[float, int, MemoryRecord]] = []
                    for score, value, record in scored:
                        extra = self._overlap_score(
                            feedback,
                            document_tokens_by_row[record.row_id],
                            feedback_df if cfg.corpus_idf else document_frequency,
                            document_count,
                        )
                        boosted.append((score + cfg.prf_weight * extra, value, record))
                    boosted.sort(
                        key=lambda item: (item[0], item[1], item[2].memory_id), reverse=True
                    )
                    scored = boosted

            if cfg.session_diversity:
                pool = scored[: max(300, requested_top_k * 3)]
                counts: Counter[str] = Counter()
                reranked: list[tuple[float, int, MemoryRecord]] = []
                remaining = list(pool)
                while remaining and len(reranked) < max(requested_top_k, 1) * 2:
                    best_index = max(
                        range(len(remaining)),
                        key=lambda i: remaining[i][0]
                        - cfg.session_diversity * counts[remaining[i][2].session_id],
                    )
                    chosen = remaining.pop(best_index)
                    counts[chosen[2].session_id] += 1
                    reranked.append(chosen)
                scored = reranked + remaining

            context_sessions: dict[str, list[MemoryRecord]] = {}
            context_positions: dict[int, int] = {}
            if self.context_window_radius:
                selected_sessions = tuple(
                    dict.fromkeys(record.session_id for _, _, record in scored)
                )
                placeholders = ",".join("?" for _ in selected_sessions)
                for row in connection.execute(
                    f"""
                    SELECT * FROM memory
                    WHERE user_id = ? AND session_id IN ({placeholders})
                    ORDER BY session_id, row_id
                    """,
                    (user, *selected_sessions),
                ):
                    record = self._record(row)
                    context_sessions.setdefault(record.session_id, []).append(record)
                for records in context_sessions.values():
                    for position, record in enumerate(records):
                        context_positions[record.row_id] = position

            def context_neighbors(center: MemoryRecord) -> list[MemoryRecord]:
                records = context_sessions[center.session_id]
                position = context_positions[center.row_id]
                start = max(0, position - self.context_window_radius)
                stop = min(len(records), position + self.context_window_radius + 1)
                return records[start:stop]

            if self.context_window_radius and self.deduplicate_context_windows:
                selected = []
                covered_memory_ids: set[str] = set()
                for item in scored:
                    center = item[2]
                    if center.memory_id in covered_memory_ids:
                        continue
                    selected.append(item)
                    for neighbor in context_neighbors(center):
                        covered_memory_ids.add(neighbor.memory_id)
                    if len(selected) == requested_top_k:
                        break
            else:
                selected = scored[:requested_top_k]

            result = []
            for result_rank, (score, _, record) in enumerate(selected, start=1):
                expand_context = self.context_window_radius and (
                    self.context_window_top_n is None
                    or result_rank <= self.context_window_top_n
                )
                if expand_context:
                    neighbors = context_neighbors(record)
                    result_id = "ctx_" + _digest(
                        {
                            "center": record.memory_id,
                            "members": [neighbor.memory_id for neighbor in neighbors],
                        },
                        length=24,
                    )
                    content = "\n\n".join(
                        f"[memory_id={neighbor.memory_id}]\n{neighbor.render(cfg.iso_dates)}"
                        for neighbor in neighbors
                    )
                else:
                    result_id = record.memory_id
                    content = record.render(cfg.iso_dates)
                result.append(
                    {
                        "id": result_id,
                        "content": content,
                        "score": round(score, 8),
                        **(
                            {
                                "created_at": datetime.fromtimestamp(
                                    record.timestamp_ms / 1000, tz=UTC
                                )
                                .isoformat()
                                .replace("+00:00", "Z")
                            }
                            if record.timestamp_ms is not None
                            else {}
                        ),
                    }
                )
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
