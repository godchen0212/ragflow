#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Unit tests for ``Mem0MemoryStore`` using ``MockMem0Memory``.
Inherits the full contract test suite.
"""
import pytest
from memory.services.mem0_store import Mem0MemoryStore
from test.unit.memory.mocks import MockMem0Memory
from test.unit.memory.test_memory_store_interface import MemoryStoreContractTests


def _create_mem0_store() -> Mem0MemoryStore:
    """Create a Mem0MemoryStore with a mock mem0 backend."""
    store = object.__new__(Mem0MemoryStore)
    store._mem0 = MockMem0Memory()
    return store


class TestMem0MemoryStore(MemoryStoreContractTests):

    @pytest.fixture
    def store(self):
        return _create_mem0_store()

    # ── mem0-specific tests ──

    def test_has_index_always_true(self):
        s = _create_mem0_store()
        assert s.has_index("any_user", "any_memory") is True

    def test_create_index_always_true(self):
        s = _create_mem0_store()
        assert s.create_index("u1", "m1", 128) is True

    def test_get_missing_field_messages_returns_empty(self):
        s = _create_mem0_store()
        result = s.get_missing_field_messages("m1", "u1", "tokenized_content_ltks")
        assert result == []

    def test_delete_index_removes_all_for_memory(self):
        s = _create_mem0_store()
        msgs = [
            {
                "message_id": 1, "message_type": "raw", "source_id": 0,
                "memory_id": "m1", "user_id": "", "agent_id": "a1",
                "session_id": "s1", "content": "hello",
                "content_embed": [0.1, 0.2], "valid_at": "2025-01-01",
                "invalid_at": None, "forget_at": None, "status": True,
            },
            {
                "message_id": 2, "message_type": "raw", "source_id": 0,
                "memory_id": "m2", "user_id": "", "agent_id": "a1",
                "session_id": "s1", "content": "world",
                "content_embed": [0.1, 0.2], "valid_at": "2025-01-01",
                "invalid_at": None, "forget_at": None, "status": True,
            },
        ]
        s.insert_message([msgs[0]], "u1", "m1")
        s.insert_message([msgs[1]], "u1", "m2")

        s.delete_index("u1", "m1")

        # m1 messages gone
        assert s.get_by_message_id("m1", 1, "u1") is None
        # m2 messages still there
        assert s.get_by_message_id("m2", 2, "u1") is not None

    def test_calculate_message_size_without_embed(self):
        s = _create_mem0_store()
        msg = {"content": "some text"}
        size = s.calculate_message_size(msg)
        assert size > 0

    def test_calculate_message_size_with_embed(self):
        s = _create_mem0_store()
        msg = {"content": "some text", "content_embed": [0.1, 0.2, 0.3]}
        size = s.calculate_message_size(msg)
        assert size > 0

    def test_search_returns_filtered_by_memory_id(self):
        s = _create_mem0_store()
        s.insert_message([
            {
                "message_id": 1, "message_type": "raw", "source_id": 0,
                "memory_id": "m1", "user_id": "", "agent_id": "a1",
                "session_id": "s1", "content": "target content",
                "content_embed": [0.1], "valid_at": "2025-01-01",
                "invalid_at": None, "forget_at": None, "status": True,
            },
        ], "u1", "m1")
        s.insert_message([
            {
                "message_id": 2, "message_type": "raw", "source_id": 0,
                "memory_id": "m2", "user_id": "", "agent_id": "a1",
                "session_id": "s1", "content": "other content",
                "content_embed": [0.1], "valid_at": "2025-01-01",
                "invalid_at": None, "forget_at": None, "status": True,
            },
        ], "u1", "m2")

        # Create a mock MatchTextExpr-like object with original_query
        class FakeExpr:
            def __init__(self, query):
                self.extra_options = {"original_query": query}
                self.matching_text = query

        results = s.search_message(["m1"], {}, ["u1"], [FakeExpr("target")], top_n=10)
        # all results should be from m1
        for r in results:
            assert r["memory_id"] == "m1"

    def test_pick_messages_forgotten_first(self):
        s = _create_mem0_store()
        s.insert_message([
            {
                "message_id": 1, "message_type": "raw", "source_id": 0,
                "memory_id": "m1", "user_id": "", "agent_id": "a1",
                "session_id": "s1", "content": "x" * 50,
                "content_embed": [0.1], "valid_at": "2025-01-01",
                "invalid_at": None, "forget_at": "2025-01-05", "status": True,
            },
            {
                "message_id": 2, "message_type": "raw", "source_id": 0,
                "memory_id": "m1", "user_id": "", "agent_id": "a1",
                "session_id": "s1", "content": "y" * 50,
                "content_embed": [0.1], "valid_at": "2025-01-02",
                "invalid_at": None, "forget_at": None, "status": True,
            },
        ], "u1", "m1")

        ids, freed = s.pick_messages_to_delete_by_fifo("m1", "u1", 1)
        assert len(ids) >= 1
        # forgotten message should be picked first
        assert ids[0] == 1
