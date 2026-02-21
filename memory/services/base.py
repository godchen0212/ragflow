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
Abstract base class for memory message storage backends.

The interface mirrors the original MessageService method signatures.
Implementations:
  - NativeMemoryStore: delegates to settings.msgStoreConn (ES/Infinity/OceanBase)
  - Mem0MemoryStore: delegates to mem0 library

Future extension: per-memory backend selection (e.g. via Memory.storage_type),
or graph-based storage backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class MemoryStoreBase(ABC):

    # ── index management ──

    @abstractmethod
    def has_index(self, uid: str, memory_id: str) -> bool:
        """Check whether the storage index for the given memory exists."""
        ...

    @abstractmethod
    def create_index(self, uid: str, memory_id: str, vector_size: int) -> bool:
        """Create a storage index. Returns True on success."""
        ...

    @abstractmethod
    def delete_index(self, uid: str, memory_id: str) -> None:
        """Delete the storage index and all its data."""
        ...

    # ── CRUD ──

    @abstractmethod
    def insert_message(self, messages: List[dict], uid: str, memory_id: str) -> List[str]:
        """
        Bulk-insert messages. Each message dict must contain at least:
          message_id (int), message_type (str), source_id (int), memory_id (str),
          user_id (str), agent_id (str), session_id (str), content (str),
          content_embed (list[float]), valid_at (str|None), invalid_at (str|None),
          forget_at (str|None), status (bool|int).
        Returns a list of error strings (empty on full success).
        """
        ...

    @abstractmethod
    def update_message(self, condition: dict, update_dict: dict, uid: str, memory_id: str) -> bool:
        """Update messages matching *condition* with *update_dict*. Returns True on success."""
        ...

    @abstractmethod
    def delete_message(self, condition: dict, uid: str, memory_id: str) -> int:
        """Delete messages matching *condition*. Returns the number of deleted messages."""
        ...

    # ── query ──

    @abstractmethod
    def list_message(
        self, uid: str, memory_id: str,
        agent_ids: List[str] | None = None,
        keywords: str | None = None,
        page: int = 1, page_size: int = 50,
    ) -> dict:
        """
        Paginated listing of RAW messages with their extracted children.
        Returns {"message_list": [...], "total_count": int}.
        """
        ...

    @abstractmethod
    def get_recent_messages(
        self, uid_list: List[str], memory_ids: List[str],
        agent_id: str, session_id: str, limit: int,
    ) -> List[dict]:
        """Return the most recent messages for a given agent/session."""
        ...

    @abstractmethod
    def search_message(
        self, memory_ids: List[str], condition_dict: dict,
        uid_list: List[str], match_expressions: list, top_n: int,
    ) -> List[dict]:
        """
        Hybrid (text + vector) search across memories.
        *match_expressions* may contain MatchTextExpr / MatchDenseExpr / FusionExpr.
        """
        ...

    @abstractmethod
    def filter_by_keywords(
        self, keywords: List[str], uid: str, memory_ids: List[str], top_n: int,
    ) -> List[dict]:
        """
        Exact keyword filtering across memories.
        Returns messages whose keywords field contains any of the provided keywords.
        """
        ...

    # ── size / eviction ──

    @abstractmethod
    def calculate_message_size(self, message: dict) -> int:
        """Return the approximate byte size of a single message."""
        ...

    @abstractmethod
    def calculate_memory_size(self, memory_ids: List[str], uid_list: List[str]) -> dict:
        """Return {memory_id: total_bytes} for the given memories."""
        ...

    @abstractmethod
    def pick_messages_to_delete_by_fifo(
        self, memory_id: str, uid: str, size_to_delete: int,
    ) -> tuple[list, int]:
        """
        Select messages to evict under FIFO policy.
        Returns (list_of_message_ids, total_freed_bytes).
        """
        ...

    # ── misc ──

    @abstractmethod
    def get_missing_field_messages(self, memory_id: str, uid: str, field_name: str) -> List[dict]:
        """Find messages missing a specific field (used for ES migration fixes)."""
        ...

    @abstractmethod
    def get_by_message_id(self, memory_id: str, message_id: int, uid: str) -> dict | None:
        """Retrieve a single message by its integer message_id."""
        ...

    @abstractmethod
    def get_max_message_id(self, uid_list: List[str], memory_ids: List[str]) -> int:
        """Return the highest message_id across the given memories (for sequence init)."""
        ...
