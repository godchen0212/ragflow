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
Tests that ``MessageService`` (the static facade) correctly delegates
every call to the active ``MemoryStoreBase`` implementation.
"""
import pytest
from unittest.mock import Mock, call

from memory.services.base import MemoryStoreBase
from memory.services.messages import MessageService, set_store, get_store, _store


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the global store before/after each test."""
    set_store(None)
    yield
    set_store(None)


@pytest.fixture
def mock_store():
    store = Mock(spec=MemoryStoreBase)
    set_store(store)
    return store


def test_get_store_raises_when_not_initialised():
    with pytest.raises(RuntimeError, match="not initialised"):
        get_store()


def test_has_index(mock_store):
    mock_store.has_index.return_value = True
    assert MessageService.has_index("u1", "m1") is True
    mock_store.has_index.assert_called_once_with("u1", "m1")


def test_create_index(mock_store):
    mock_store.create_index.return_value = True
    assert MessageService.create_index("u1", "m1", 128) is True
    mock_store.create_index.assert_called_once_with("u1", "m1", 128)


def test_delete_index(mock_store):
    MessageService.delete_index("u1", "m1")
    mock_store.delete_index.assert_called_once_with("u1", "m1")


def test_insert_message(mock_store):
    mock_store.insert_message.return_value = []
    msgs = [{"message_id": 1}]
    result = MessageService.insert_message(msgs, "u1", "m1")
    assert result == []
    mock_store.insert_message.assert_called_once_with(msgs, "u1", "m1")


def test_update_message(mock_store):
    mock_store.update_message.return_value = True
    cond = {"message_id": 1}
    upd = {"status": False}
    assert MessageService.update_message(cond, upd, "u1", "m1") is True
    mock_store.update_message.assert_called_once_with(cond, upd, "u1", "m1")


def test_delete_message(mock_store):
    mock_store.delete_message.return_value = 1
    assert MessageService.delete_message({"message_id": [1]}, "u1", "m1") == 1
    mock_store.delete_message.assert_called_once_with({"message_id": [1]}, "u1", "m1")


def test_list_message(mock_store):
    mock_store.list_message.return_value = {"message_list": [], "total_count": 0}
    result = MessageService.list_message("u1", "m1", agent_ids=["a1"], page=2, page_size=10)
    assert result == {"message_list": [], "total_count": 0}
    mock_store.list_message.assert_called_once_with("u1", "m1", ["a1"], None, 2, 10)


def test_get_recent_messages(mock_store):
    mock_store.get_recent_messages.return_value = [{"message_id": 1}]
    result = MessageService.get_recent_messages(["u1"], ["m1"], "a1", "s1", 5)
    assert len(result) == 1
    mock_store.get_recent_messages.assert_called_once_with(["u1"], ["m1"], "a1", "s1", 5)


def test_search_message(mock_store):
    mock_store.search_message.return_value = []
    result = MessageService.search_message(["m1"], {}, ["u1"], [], 10)
    assert result == []
    mock_store.search_message.assert_called_once_with(["m1"], {}, ["u1"], [], 10)


def test_calculate_message_size(mock_store):
    mock_store.calculate_message_size.return_value = 42
    msg = {"content": "hi", "content_embed": [0.1]}
    assert MessageService.calculate_message_size(msg) == 42
    mock_store.calculate_message_size.assert_called_once_with(msg)


def test_calculate_memory_size(mock_store):
    mock_store.calculate_memory_size.return_value = {"m1": 100}
    result = MessageService.calculate_memory_size(["m1"], ["u1"])
    assert result == {"m1": 100}
    mock_store.calculate_memory_size.assert_called_once_with(["m1"], ["u1"])


def test_pick_messages_to_delete_by_fifo(mock_store):
    mock_store.pick_messages_to_delete_by_fifo.return_value = ([1, 2], 200)
    ids, freed = MessageService.pick_messages_to_delete_by_fifo("m1", "u1", 100)
    assert ids == [1, 2]
    assert freed == 200
    mock_store.pick_messages_to_delete_by_fifo.assert_called_once_with("m1", "u1", 100)


def test_get_missing_field_messages(mock_store):
    mock_store.get_missing_field_messages.return_value = []
    result = MessageService.get_missing_field_messages("m1", "u1", "some_field")
    assert result == []
    mock_store.get_missing_field_messages.assert_called_once_with("m1", "u1", "some_field")


def test_get_by_message_id(mock_store):
    mock_store.get_by_message_id.return_value = {"message_id": 5}
    result = MessageService.get_by_message_id("m1", 5, "u1")
    assert result == {"message_id": 5}
    mock_store.get_by_message_id.assert_called_once_with("m1", 5, "u1")


def test_get_max_message_id(mock_store):
    mock_store.get_max_message_id.return_value = 99
    assert MessageService.get_max_message_id(["u1"], ["m1"]) == 99
    mock_store.get_max_message_id.assert_called_once_with(["u1"], ["m1"])
