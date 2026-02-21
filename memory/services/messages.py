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
Backward-compatible static facade for memory message storage.

All callers continue to use ``MessageService.method(...)`` unchanged.
Under the hood every call is delegated to the active ``MemoryStoreBase``
implementation selected during ``init_settings()``.
"""
from __future__ import annotations

from typing import List, Optional

from memory.services.base import MemoryStoreBase

_store: MemoryStoreBase | None = None


def get_store() -> MemoryStoreBase:
    """Return the active memory store. Raises if not yet initialised."""
    if _store is None:
        raise RuntimeError(
            "Memory store not initialised. "
            "Call common.settings.init_settings() first."
        )
    return _store


def set_store(store: MemoryStoreBase) -> None:
    """Set the global memory store instance (called from init_settings)."""
    global _store
    _store = store


class MessageService:
    """Static facade — every classmethod delegates to ``get_store()``."""

    @classmethod
    def has_index(cls, uid: str, memory_id: str) -> bool:
        return get_store().has_index(uid, memory_id)

    @classmethod
    def create_index(cls, uid: str, memory_id: str, vector_size: int) -> bool:
        return get_store().create_index(uid, memory_id, vector_size)

    @classmethod
    def delete_index(cls, uid: str, memory_id: str) -> None:
        return get_store().delete_index(uid, memory_id)

    @classmethod
    def insert_message(cls, messages: List[dict], uid: str, memory_id: str) -> List[str]:
        return get_store().insert_message(messages, uid, memory_id)

    @classmethod
    def update_message(cls, condition: dict, update_dict: dict, uid: str, memory_id: str) -> bool:
        return get_store().update_message(condition, update_dict, uid, memory_id)

    @classmethod
    def delete_message(cls, condition: dict, uid: str, memory_id: str) -> int:
        return get_store().delete_message(condition, uid, memory_id)

    @classmethod
    def list_message(cls, uid: str, memory_id: str, agent_ids: List[str] = None,
                     keywords: str = None, page: int = 1, page_size: int = 50) -> dict:
        return get_store().list_message(uid, memory_id, agent_ids, keywords, page, page_size)

    @classmethod
    def get_recent_messages(cls, uid_list: List[str], memory_ids: List[str],
                            agent_id: str, session_id: str, limit: int) -> List[dict]:
        return get_store().get_recent_messages(uid_list, memory_ids, agent_id, session_id, limit)

    @classmethod
    def search_message(cls, memory_ids: List[str], condition_dict: dict,
                       uid_list: List[str], match_expressions: list, top_n: int) -> List[dict]:
        return get_store().search_message(memory_ids, condition_dict, uid_list, match_expressions, top_n)

    @classmethod
    def filter_by_keywords(cls, keywords: List[str], uid: str, memory_ids: List[str], top_n: int) -> List[dict]:
        return get_store().filter_by_keywords(keywords, uid, memory_ids, top_n)

    @staticmethod
    def calculate_message_size(message: dict) -> int:
        return get_store().calculate_message_size(message)

    @classmethod
    def calculate_memory_size(cls, memory_ids: List[str], uid_list: List[str]) -> dict:
        return get_store().calculate_memory_size(memory_ids, uid_list)

    @classmethod
    def pick_messages_to_delete_by_fifo(cls, memory_id: str, uid: str, size_to_delete: int) -> tuple[list, int]:
        return get_store().pick_messages_to_delete_by_fifo(memory_id, uid, size_to_delete)

    @classmethod
    def get_missing_field_messages(cls, memory_id: str, uid: str, field_name: str) -> List[dict]:
        return get_store().get_missing_field_messages(memory_id, uid, field_name)

    @classmethod
    def get_by_message_id(cls, memory_id: str, message_id: int, uid: str) -> dict | None:
        return get_store().get_by_message_id(memory_id, message_id, uid)

    @classmethod
    def get_max_message_id(cls, uid_list: List[str], memory_ids: List[str]) -> int:
        return get_store().get_max_message_id(uid_list, memory_ids)
