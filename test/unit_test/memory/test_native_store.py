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
Unit tests for ``NativeMemoryStore`` using ``MockMsgStoreConn``.
Inherits the full contract test suite.
"""
import pytest
from memory.services.native_store import NativeMemoryStore
from test.unit.memory.mocks import MockMsgStoreConn
from test.unit.memory.test_memory_store_interface import MemoryStoreContractTests


class TestNativeMemoryStore(MemoryStoreContractTests):

    @pytest.fixture
    def store(self):
        return NativeMemoryStore(MockMsgStoreConn())

    # ── native-specific tests ──

    def test_index_name_format(self):
        s = NativeMemoryStore(MockMsgStoreConn())
        assert s._index_name("user123") == "memory_user123"

    def test_insert_sets_composite_id(self):
        conn = MockMsgStoreConn()
        s = NativeMemoryStore(conn)
        s.create_index("u1", "m1", 4)
        msg = {
            "message_id": 42,
            "message_type": "raw",
            "source_id": 0,
            "memory_id": "m1",
            "user_id": "",
            "agent_id": "a1",
            "session_id": "s1",
            "content": "test",
            "content_embed": [0.1, 0.2, 0.3, 0.4],
            "valid_at": "2025-01-01",
            "invalid_at": None,
            "forget_at": None,
            "status": True,
        }
        s.insert_message([msg], "u1", "m1")
        # the doc should be stored with composite id "m1_42"
        raw = conn.get("m1_42", "memory_u1", ["m1"])
        assert raw is not None

    def test_forgotten_messages_picked_first(self):
        conn = MockMsgStoreConn()
        s = NativeMemoryStore(conn)
        s.create_index("u1", "m1", 4)

        def _msg(mid, forget_at=None, valid_at="2025-01-01"):
            return {
                "message_id": mid,
                "message_type": "raw",
                "source_id": 0,
                "memory_id": "m1",
                "user_id": "",
                "agent_id": "a1",
                "session_id": "s1",
                "content": "x" * 50,
                "content_embed": [0.1, 0.2, 0.3, 0.4],
                "valid_at": valid_at,
                "invalid_at": None,
                "forget_at": forget_at,
                "status": True,
            }

        s.insert_message([
            _msg(1, forget_at="2025-01-05", valid_at="2025-01-01"),
            _msg(2, valid_at="2025-01-02"),
            _msg(3, valid_at="2025-01-03"),
        ], "u1", "m1")

        ids, _ = s.pick_messages_to_delete_by_fifo("m1", "u1", 1)
        # forgotten message (id=1) should be picked first
        assert ids[0] == 1
