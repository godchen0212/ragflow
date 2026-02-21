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
Contract tests for ``MemoryStoreBase`` implementations.

Subclass this and provide a ``store`` fixture to run the full suite
against any backend.
"""
import pytest
from typing import Optional, List
from memory.services.base import MemoryStoreBase

UID = "test_user_001"
MEMORY_ID = "mem_001"
VECTOR_SIZE = 4


def _make_message(
    message_id: int,
    message_type: str = "raw",
    source_id: int = 0,
    memory_id: str = MEMORY_ID,
    agent_id: str = "agent_1",
    session_id: str = "sess_1",
    content: str = "hello world",
    valid_at: str = "2025-01-01 00:00:00",
    status: bool = True,
    keywords: Optional[List[str]] = None,
) -> dict:
    return {
        "message_id": message_id,
        "message_type": message_type,
        "source_id": source_id,
        "memory_id": memory_id,
        "user_id": "",
        "agent_id": agent_id,
        "session_id": session_id,
        "content": content,
        "content_embed": [0.1] * VECTOR_SIZE,
        "valid_at": valid_at,
        "invalid_at": None,
        "forget_at": None,
        "status": status,
        "zone_id": 0,
        "keywords": keywords or [],
    }


class MemoryStoreContractTests:
    """Mixin — subclass and supply a ``store`` fixture."""

    # ── index management ──

    def test_create_and_has_index(self, store: MemoryStoreBase):
        assert store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        assert store.has_index(UID, MEMORY_ID)

    def test_delete_index(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message([_make_message(1)], UID, MEMORY_ID)
        store.delete_index(UID, MEMORY_ID)
        # after deletion, get_by_message_id should return None
        assert store.get_by_message_id(MEMORY_ID, 1, UID) is None

    # ── insert / get ──

    def test_insert_and_get_by_message_id(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        msg = _make_message(1, content="test content")
        errors = store.insert_message([msg], UID, MEMORY_ID)
        assert errors == []

        result = store.get_by_message_id(MEMORY_ID, 1, UID)
        assert result is not None
        assert result["message_id"] == 1
        assert "test content" in result.get("content", "")

    def test_insert_multiple(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        msgs = [_make_message(i, content=f"msg {i}") for i in range(1, 4)]
        errors = store.insert_message(msgs, UID, MEMORY_ID)
        assert errors == []

        for i in range(1, 4):
            r = store.get_by_message_id(MEMORY_ID, i, UID)
            assert r is not None

    # ── update ──

    def test_update_message(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message([_make_message(1, status=True)], UID, MEMORY_ID)

        ok = store.update_message(
            {"message_id": 1}, {"status": False}, UID, MEMORY_ID
        )
        assert ok is True

    # ── delete ──

    def test_delete_message(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message([_make_message(1), _make_message(2)], UID, MEMORY_ID)

        deleted = store.delete_message({"message_id": [1]}, UID, MEMORY_ID)
        assert deleted >= 1

        assert store.get_by_message_id(MEMORY_ID, 1, UID) is None
        assert store.get_by_message_id(MEMORY_ID, 2, UID) is not None

    # ── list ──

    def test_list_message_pagination(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        # insert 3 raw + 2 extracted
        msgs = [
            _make_message(1, message_type="raw", valid_at="2025-01-01 00:00:00"),
            _make_message(2, message_type="raw", valid_at="2025-01-02 00:00:00"),
            _make_message(3, message_type="raw", valid_at="2025-01-03 00:00:00"),
            _make_message(4, message_type="semantic", source_id=1, valid_at="2025-01-01 00:00:01"),
            _make_message(5, message_type="episodic", source_id=2, valid_at="2025-01-02 00:00:01"),
        ]
        store.insert_message(msgs, UID, MEMORY_ID)

        result = store.list_message(UID, MEMORY_ID, page=1, page_size=2)
        assert "message_list" in result
        assert "total_count" in result
        assert result["total_count"] == 3  # 3 raw messages
        assert len(result["message_list"]) == 2  # page_size=2

    def test_list_message_with_extracts(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        msgs = [
            _make_message(1, message_type="raw"),
            _make_message(2, message_type="semantic", source_id=1),
        ]
        store.insert_message(msgs, UID, MEMORY_ID)

        result = store.list_message(UID, MEMORY_ID)
        assert result["total_count"] == 1
        raw_msg = result["message_list"][0]
        assert "extract" in raw_msg
        assert len(raw_msg["extract"]) == 1

    # ── recent messages ──

    def test_get_recent_messages_ordering(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        msgs = [
            _make_message(1, agent_id="a1", session_id="s1", valid_at="2025-01-01 00:00:00", content="old"),
            _make_message(2, agent_id="a1", session_id="s1", valid_at="2025-01-03 00:00:00", content="new"),
            _make_message(3, agent_id="a1", session_id="s1", valid_at="2025-01-02 00:00:00", content="mid"),
        ]
        store.insert_message(msgs, UID, MEMORY_ID)

        recent = store.get_recent_messages([UID], [MEMORY_ID], "a1", "s1", limit=2)
        assert len(recent) == 2
        # most recent first
        assert recent[0]["valid_at"] >= recent[1]["valid_at"]

    # ── search ──

    def test_search_message_returns_list(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [_make_message(1, content="the quick brown fox")],
            UID, MEMORY_ID,
        )
        # search with empty match_expressions — should still return results
        results = store.search_message(
            [MEMORY_ID], {}, [UID], match_expressions=[], top_n=10,
        )
        assert isinstance(results, list)

    # ── size / eviction ──

    def test_calculate_message_size(self, store: MemoryStoreBase):
        msg = _make_message(1, content="some content")
        size = store.calculate_message_size(msg)
        assert size > 0

    def test_calculate_memory_size(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [_make_message(1, content="abc"), _make_message(2, content="def")],
            UID, MEMORY_ID,
        )
        size_map = store.calculate_memory_size([MEMORY_ID], [UID])
        assert MEMORY_ID in size_map
        assert size_map[MEMORY_ID] > 0

    def test_pick_messages_to_delete_by_fifo(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [
                _make_message(1, content="a" * 100, valid_at="2025-01-01 00:00:00"),
                _make_message(2, content="b" * 100, valid_at="2025-01-02 00:00:00"),
            ],
            UID, MEMORY_ID,
        )
        ids, freed = store.pick_messages_to_delete_by_fifo(MEMORY_ID, UID, 1)
        assert len(ids) >= 1
        assert freed > 0

    # ── query: keywords filtering ──

    def test_filter_by_keywords_single_keyword(self, store: MemoryStoreBase):
        """Test filtering with a single keyword."""
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [
                _make_message(1, content="python tutorial", keywords=["python", "tutorial"]),
                _make_message(2, content="java guide", keywords=["java"]),
                _make_message(3, content="python advanced", keywords=["python", "advanced"]),
            ],
            UID, MEMORY_ID,
        )
        results = store.filter_by_keywords(["python"], UID, [MEMORY_ID], top_n=10)
        assert len(results) == 2
        assert all(r["message_id"] in [1, 3] for r in results)

    def test_filter_by_keywords_multiple_keywords_or_logic(self, store: MemoryStoreBase):
        """Test filtering with multiple keywords (OR logic)."""
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [
                _make_message(1, content="python", keywords=["python"]),
                _make_message(2, content="java", keywords=["java"]),
                _make_message(3, content="rust", keywords=["rust"]),
            ],
            UID, MEMORY_ID,
        )
        results = store.filter_by_keywords(["python", "java"], UID, [MEMORY_ID], top_n=10)
        assert len(results) == 2
        assert all(r["message_id"] in [1, 2] for r in results)

    def test_filter_by_keywords_no_match(self, store: MemoryStoreBase):
        """Test filtering returns empty list when no keywords match."""
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [_make_message(1, content="hello", keywords=["greeting"])],
            UID, MEMORY_ID,
        )
        results = store.filter_by_keywords(["nonexistent"], UID, [MEMORY_ID], top_n=10)
        assert len(results) == 0

    def test_filter_by_keywords_empty_keywords_field(self, store: MemoryStoreBase):
        """Test filtering with messages that have empty keywords."""
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [
                _make_message(1, content="no keywords", keywords=[]),
                _make_message(2, content="has keywords", keywords=["test"]),
            ],
            UID, MEMORY_ID,
        )
        results = store.filter_by_keywords(["test"], UID, [MEMORY_ID], top_n=10)
        assert len(results) == 1
        assert results[0]["message_id"] == 2

    def test_filter_by_keywords_empty_filter_list(self, store: MemoryStoreBase):
        """Test filtering with empty keyword list returns empty result."""
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [_make_message(1, content="test", keywords=["test"])],
            UID, MEMORY_ID,
        )
        results = store.filter_by_keywords([], UID, [MEMORY_ID], top_n=10)
        assert len(results) == 0

    def test_filter_by_keywords_respects_top_n(self, store: MemoryStoreBase):
        """Test that filter_by_keywords respects the top_n limit."""
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [
                _make_message(1, content="a", keywords=["tag"], valid_at="2025-01-01 00:00:00"),
                _make_message(2, content="b", keywords=["tag"], valid_at="2025-01-02 00:00:00"),
                _make_message(3, content="c", keywords=["tag"], valid_at="2025-01-03 00:00:00"),
            ],
            UID, MEMORY_ID,
        )
        results = store.filter_by_keywords(["tag"], UID, [MEMORY_ID], top_n=2)
        assert len(results) == 2

    # ── misc ──

    def test_get_max_message_id(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        store.insert_message(
            [_make_message(5), _make_message(10), _make_message(3)],
            UID, MEMORY_ID,
        )
        max_id = store.get_max_message_id([UID], [MEMORY_ID])
        assert max_id == 10

    def test_get_max_message_id_empty(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        max_id = store.get_max_message_id([UID], [MEMORY_ID])
        assert max_id == 1  # default when empty

    def test_get_missing_field_messages(self, store: MemoryStoreBase):
        store.create_index(UID, MEMORY_ID, VECTOR_SIZE)
        # this is mainly an ES-specific operation; just verify it returns a list
        result = store.get_missing_field_messages(MEMORY_ID, UID, "tokenized_content_ltks")
        assert isinstance(result, list)
