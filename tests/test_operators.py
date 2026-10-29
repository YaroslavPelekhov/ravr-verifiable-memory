from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ravr_science.memory import OperatorConfig, VerifiableMemoryStore


def _fill(store: VerifiableMemoryStore, user: str = "u") -> None:
    store.add(
        request_id=f"{user}-1",
        user_id=user,
        session_id="s1",
        messages=[
            {"role": "user", "timestamp": 1_700_000_000_000, "content": "Maria lives in Porto."},
            {"role": "assistant", "timestamp": 1_700_000_001_000, "content": "Porto is lovely."},
            {"role": "user", "timestamp": 1_700_000_002_000, "content": "I love ceramic art."},
        ],
    )
    store.add(
        request_id=f"{user}-2",
        user_id=user,
        session_id="s2",
        messages=[
            {"role": "user", "timestamp": 1_700_100_000_000, "content": "Maria moved to Oslo."},
            {"role": "assistant", "timestamp": 1_700_100_001_000, "content": "Oslo is cold."},
        ],
    )


class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "m.sqlite3"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _store(self, **operators) -> VerifiableMemoryStore:
        return VerifiableMemoryStore(
            self.path, context_window_radius=0, operators=OperatorConfig(**operators)
        )

    def test_default_operators_change_nothing(self) -> None:
        store = self._store()
        _fill(store)
        explicit = VerifiableMemoryStore(
            Path(self.directory.name) / "other.sqlite3", context_window_radius=0
        )
        _fill(explicit)
        query = {"query": "Where does Maria live?", "user_id": "u", "top_k": 5}
        self.assertEqual(
            [item["id"] for item in store.search(**query)["data"]],
            [item["id"] for item in explicit.search(**query)["data"]],
        )

    def test_stemming_matches_inflected_forms(self) -> None:
        store = self._store(stem=True, corpus_idf=True)
        _fill(store)
        top = store.search(query="Who is living in Porto?", user_id="u", top_k=1)["data"][0]
        self.assertIn("lives in Porto", top["content"])

    def test_term_df_tracks_each_user_separately(self) -> None:
        store = self._store(corpus_idf=True)
        _fill(store, "alice")
        _fill(store, "bob")
        with store._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, df FROM term_df WHERE token = 'maria' ORDER BY user_id"
            ).fetchall()
        self.assertEqual([(row["user_id"], row["df"]) for row in rows], [("alice", 2), ("bob", 2)])

    def test_role_user_breaks_ties_toward_user_turns(self) -> None:
        store = self._store(role_user=0.5)
        store.add(
            request_id="r",
            user_id="u",
            session_id="s",
            messages=[
                {"role": "assistant", "content": "The picnic is on Saturday."},
                {"role": "user", "content": "The picnic is on Saturday."},
            ],
        )
        top = store.search(query="When is the picnic?", user_id="u", top_k=1)["data"][0]
        self.assertIn("role=user", top["content"])

    def test_iso_dates_are_added_without_removing_the_timestamp(self) -> None:
        store = self._store(iso_dates=True)
        _fill(store)
        content = store.search(query="Where does Maria live?", user_id="u", top_k=1)["data"][0][
            "content"
        ]
        self.assertIn("timestamp=1700100000000", content)
        self.assertIn("date=2023-11-16 Thu 02:00 UTC", content)

    def test_passthrough_returns_the_whole_small_memory_in_order(self) -> None:
        store = self._store(passthrough_chars=10_000)
        _fill(store)
        data = store.search(query="anything about cats", user_id="u", top_k=100)["data"]
        self.assertEqual(len(data), 5)
        stamps = [item["created_at"] for item in data]
        self.assertEqual(stamps, sorted(stamps))

    def test_passthrough_is_skipped_when_memory_exceeds_top_k(self) -> None:
        store = self._store(passthrough_chars=10_000)
        _fill(store)
        data = store.search(query="Where does Maria live?", user_id="u", top_k=2)["data"]
        self.assertEqual(len(data), 2)
        self.assertNotEqual(data[0]["score"], 1.0)

    def test_session_diversity_spreads_results_across_sessions(self) -> None:
        store = self._store(session_diversity=1.0)
        _fill(store)
        data = store.search(query="Maria", user_id="u", top_k=2)["data"]
        text = " ".join(item["content"] for item in data)
        self.assertIn("Porto", text)
        self.assertIn("Oslo", text)

    def test_empty_turns_are_kept_as_placeholders_but_wrong_types_still_fail(self) -> None:
        store = self._store()
        store.add(
            request_id="r",
            user_id="u",
            session_id="s",
            messages=[
                {"role": "user", "content": "Mira likes tea."},
                {"role": "assistant", "content": "   "},
                {"role": "user", "content": "Mira likes jazz."},
            ],
        )
        self.assertEqual(store.stats()["memories"], 3)
        with self.assertRaises(TypeError):
            store.add(
                request_id="r2", user_id="u", session_id="s",
                messages=[{"role": "user", "content": 5}],
            )

    def test_pseudo_relevance_feedback_and_supersession_run(self) -> None:
        store = self._store(prf_terms=5, prf_weight=0.1, supersession=0.1, session_prior=0.1,
                            pref_cue=0.1, role_assistant=0.1)
        _fill(store)
        data = store.search(query="Where does Maria live now?", user_id="u", top_k=3)["data"]
        self.assertIn("Oslo", data[0]["content"])


if __name__ == "__main__":
    unittest.main()
