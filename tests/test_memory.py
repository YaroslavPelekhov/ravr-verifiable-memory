from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ravr_science.memory import VerifiableMemoryStore


class VerifiableMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = VerifiableMemoryStore(Path(self.directory.name) / "memory.sqlite3")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_add_is_idempotent_and_search_contract_is_bounded(self) -> None:
        payload = {
            "request_id": "request-1",
            "messages": [
                {"role": "user", "timestamp": 1000, "content": "Mira moved to Lisbon."},
                {"role": "assistant", "timestamp": 1001, "content": "She likes ceramic art."},
            ],
            "user_id": "user-1",
            "session_id": "session-1",
        }
        expected = {
            "success": True,
            "request_id": "request-1",
            "user_id": "user-1",
            "session_id": "session-1",
        }
        self.assertEqual(self.store.add(**payload), expected)
        self.assertEqual(self.store.add(**payload), expected)
        result = self.store.search(query="Where did Mira move?", user_id="user-1", top_k=1)
        self.assertEqual(len(result["data"]), 1)
        self.assertIn("Mira moved to Lisbon", result["data"][0]["content"])
        self.assertEqual(self.store.stats()["memories"], 2)

    def test_reusing_request_id_with_different_payload_is_rejected(self) -> None:
        common = {"request_id": "same", "user_id": "u", "session_id": "s"}
        self.store.add(messages=[{"role": "user", "content": "one"}], **common)
        with self.assertRaisesRegex(ValueError, "different payload"):
            self.store.add(messages=[{"role": "user", "content": "two"}], **common)

    def test_user_scopes_are_isolated(self) -> None:
        self.store.add(
            request_id="r-a",
            messages=[{"role": "user", "content": "The launch code is apricot."}],
            user_id="alice",
            session_id="s-a",
        )
        self.store.add(
            request_id="r-b",
            messages=[{"role": "user", "content": "The launch code is cobalt."}],
            user_id="bob",
            session_id="s-b",
        )
        result = self.store.search(query="What is the launch code?", user_id="alice", top_k=10)
        text = "\n".join(item["content"] for item in result["data"])
        self.assertIn("apricot", text)
        self.assertNotIn("cobalt", text)

    def test_temporal_question_prefers_latest_supported_memory(self) -> None:
        self.store.add(
            request_id="old",
            messages=[{"role": "user", "timestamp": 1000, "content": "Nia lives in Rome."}],
            user_id="u",
            session_id="s1",
        )
        self.store.add(
            request_id="new",
            messages=[{"role": "user", "timestamp": 2000, "content": "Nia now lives in Oslo."}],
            user_id="u",
            session_id="s2",
        )
        result = self.store.search(query="Where does Nia currently live?", user_id="u", top_k=2)
        self.assertIn("Oslo", result["data"][0]["content"])

    def test_options_expand_evidence_retrieval_without_generating_an_answer(self) -> None:
        self.store.add(
            request_id="r1",
            messages=[
                {"role": "user", "content": "For field trips, Sam always chooses archaeology."},
                {"role": "user", "content": "The group discussed music and painting."},
            ],
            user_id="u",
            session_id="s",
        )
        result = self.store.search(
            query="Which activity best matches Sam's preference?",
            options=["A. Painting", "B. Archaeology", "C. Music"],
            user_id="u",
            top_k=2,
        )
        self.assertTrue(any("archaeology" in item["content"].casefold() for item in result["data"]))
        self.assertTrue(all("final answer" not in item["content"].casefold() for item in result["data"]))

    def test_invalid_top_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            self.store.search(query="anything", user_id="u", top_k=101)


if __name__ == "__main__":
    unittest.main()
