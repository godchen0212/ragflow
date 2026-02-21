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
mem0-backed memory store implementation.

Uses the ``mem0ai`` library as the storage / retrieval engine.
RAGFlow message schema fields are persisted in mem0's ``metadata`` dict
so that the full message structure can be reconstructed on read.

Vector store defaults to Elasticsearch (reuses the ES instance already
deployed alongside RAGFlow).

Known limitations vs. the native store:
  - search_message uses mem0's pure-vector search; hybrid BM25+kNN is not
    available.  The ``match_expressions`` parameter is used only to extract
    the original query text.
  - list_message / calculate_memory_size rely on ``get_all`` + in-memory
    filtering, which may be slow for very large memories.
  - get_missing_field_messages is a no-op (returns []).
"""
from __future__ import annotations

import logging
import sys
from typing import List

from memory.services.base import MemoryStoreBase

logger = logging.getLogger(__name__)

# Message fields stored in mem0 metadata
_META_FIELDS = (
    "message_id", "message_type", "source_id", "memory_id",
    "user_id", "agent_id", "session_id",
    "valid_at", "invalid_at", "forget_at", "status", "zone_id", "keywords",
)


def _msg_to_metadata(msg: dict) -> dict:
    """Extract RAGFlow message fields into a flat metadata dict for mem0."""
    import json
    meta = {}
    for f in _META_FIELDS:
        v = msg.get(f)
        if v is None:
            meta[f] = ""
        elif f == "keywords":
            # Serialize keywords list to JSON string
            meta[f] = json.dumps(v) if isinstance(v, list) else (v or "[]")
        elif isinstance(v, bool):
            meta[f] = 1 if v else 0
        else:
            meta[f] = v
    return meta


def _metadata_to_msg(mem0_record: dict) -> dict:
    """Reconstruct a RAGFlow message dict from a mem0 record."""
    import json
    meta = mem0_record.get("metadata", {})
    kw = meta.get("keywords", "[]")
    return {
        "message_id": meta.get("message_id", 0),
        "message_type": meta.get("message_type", ""),
        "source_id": meta.get("source_id", 0),
        "memory_id": meta.get("memory_id", ""),
        "user_id": meta.get("user_id", ""),
        "agent_id": meta.get("agent_id", ""),
        "session_id": meta.get("session_id", ""),
        "valid_at": meta.get("valid_at", ""),
        "invalid_at": meta.get("invalid_at", "") or None,
        "forget_at": meta.get("forget_at", "") or None,
        "status": bool(int(meta.get("status", 1))),
        "content": mem0_record.get("memory", ""),
        "zone_id": meta.get("zone_id", 0),
        "keywords": json.loads(kw) if isinstance(kw, str) else (kw or []),
        # mem0 manages its own embeddings; we don't expose content_embed
        "_mem0_id": mem0_record.get("id", ""),
    }


def _extract_query_text(match_expressions: list) -> str:
    """
    Pull the original query string out of match_expressions.
    The native caller puts it in MatchTextExpr.extra_options["original_query"]
    or MatchDenseExpr.extra_options["original_query"].
    """
    for expr in match_expressions:
        opts = getattr(expr, "extra_options", None) or {}
        if "original_query" in opts:
            return opts["original_query"]
        # fallback: MatchTextExpr.matching_text
        if hasattr(expr, "matching_text") and expr.matching_text:
            return expr.matching_text
    return ""


class Mem0MemoryStore(MemoryStoreBase):

    def __init__(self, config: dict):
        """
        *config* is passed directly to ``mem0.Memory.from_config()``.
        Example::

            {
                "vector_store": {
                    "provider": "elasticsearch",
                    "config": {
                        "url": "http://es01:9200",
                        "collection_name": "ragflow_mem0",
                        "embedding_dims": 1536
                    }
                }
            }
        """
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError(
                "mem0ai is required for Mem0MemoryStore. "
                "Install it with: pip install mem0ai"
            )
        self._mem0 = Memory.from_config(config)

    # ── helpers ──

    def _get_all_for_memory(self, uid: str, memory_id: str) -> list[dict]:
        """Fetch all mem0 records belonging to a specific RAGFlow memory."""
        all_records = self._mem0.get_all(user_id=uid) or []
        return [r for r in all_records if r.get("metadata", {}).get("memory_id") == memory_id]

    def _get_all_for_memories(self, uid_list: list[str], memory_ids: list[str]) -> list[dict]:
        """Fetch mem0 records across multiple users/memories."""
        results = []
        seen_uids = set()
        for uid in uid_list:
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            all_records = self._mem0.get_all(user_id=uid) or []
            for r in all_records:
                if r.get("metadata", {}).get("memory_id") in memory_ids:
                    results.append(r)
        return results

    # ── index management ──

    def has_index(self, uid: str, memory_id: str) -> bool:
        # mem0 auto-manages its storage; always report as existing
        return True

    def create_index(self, uid: str, memory_id: str, vector_size: int) -> bool:
        return True

    def delete_index(self, uid: str, memory_id: str) -> None:
        records = self._get_all_for_memory(uid, memory_id)
        for r in records:
            try:
                self._mem0.delete(r["id"])
            except Exception:
                logger.warning("Failed to delete mem0 record %s", r.get("id"))

    # ── CRUD ──

    def insert_message(self, messages: List[dict], uid: str, memory_id: str) -> List[str]:
        errors: list[str] = []
        for msg in messages:
            meta = _msg_to_metadata(msg)
            meta["memory_id"] = memory_id
            content = msg.get("content", "")
            try:
                self._mem0.add(
                    content,
                    user_id=uid,
                    metadata=meta,
                )
            except Exception as e:
                err = f"mem0 insert failed for message_id={msg.get('message_id')}: {e}"
                logger.error(err)
                errors.append(err)
        return errors

    def update_message(self, condition: dict, update_dict: dict, uid: str, memory_id: str) -> bool:
        records = self._get_all_for_memory(uid, memory_id)
        matched = self._filter_by_condition(records, condition)
        if not matched:
            return False

        for r in matched:
            new_meta = dict(r.get("metadata", {}))
            new_content = r.get("memory", "")
            for k, v in update_dict.items():
                if k == "content":
                    new_content = v
                elif k == "status":
                    new_meta["status"] = 1 if v else 0
                elif k in _META_FIELDS:
                    new_meta[k] = v
            try:
                self._mem0.update(r["id"], data=new_content, metadata=new_meta)
            except Exception as e:
                logger.error("mem0 update failed for %s: %s", r.get("id"), e)
                return False
        return True

    def delete_message(self, condition: dict, uid: str, memory_id: str) -> int:
        records = self._get_all_for_memory(uid, memory_id)
        matched = self._filter_by_condition(records, condition)
        deleted = 0
        for r in matched:
            try:
                self._mem0.delete(r["id"])
                deleted += 1
            except Exception as e:
                logger.warning("mem0 delete failed for %s: %s", r.get("id"), e)
        return deleted

    # ── query ──

    def list_message(
        self, uid: str, memory_id: str,
        agent_ids: List[str] | None = None,
        keywords: str | None = None,
        page: int = 1, page_size: int = 50,
    ) -> dict:
        records = self._get_all_for_memory(uid, memory_id)

        # filter
        if agent_ids:
            records = [r for r in records if r.get("metadata", {}).get("agent_id") in agent_ids]
        if keywords:
            records = [r for r in records if keywords in str(r.get("metadata", {}).get("session_id", ""))]

        # separate raw vs extracted
        raw_records = [r for r in records if r.get("metadata", {}).get("message_type") == "raw"]
        extract_records = [r for r in records if r.get("metadata", {}).get("message_type") != "raw"]

        # sort raw by valid_at desc
        raw_records.sort(key=lambda r: r.get("metadata", {}).get("valid_at", ""), reverse=True)
        total_count = len(raw_records)

        # paginate
        start = (page - 1) * page_size
        page_records = raw_records[start:start + page_size]

        # group extracts by source_id
        extract_by_source: dict = {}
        for r in extract_records:
            sid = r.get("metadata", {}).get("source_id", 0)
            extract_by_source.setdefault(sid, []).append(_metadata_to_msg(r))

        message_list = []
        for r in page_records:
            msg = _metadata_to_msg(r)
            msg["extract"] = extract_by_source.get(msg["message_id"], [])
            message_list.append(msg)

        return {"message_list": message_list, "total_count": total_count}

    def get_recent_messages(
        self, uid_list: List[str], memory_ids: List[str],
        agent_id: str, session_id: str, limit: int,
    ) -> List[dict]:
        records = self._get_all_for_memories(uid_list, memory_ids)

        # filter by agent_id and session_id
        filtered = []
        for r in records:
            meta = r.get("metadata", {})
            if meta.get("agent_id") == agent_id and meta.get("session_id") == session_id:
                filtered.append(r)

        # sort by valid_at desc, take limit
        filtered.sort(key=lambda r: r.get("metadata", {}).get("valid_at", ""), reverse=True)
        return [_metadata_to_msg(r) for r in filtered[:limit]]

    def search_message(
        self, memory_ids: List[str], condition_dict: dict,
        uid_list: List[str], match_expressions: list, top_n: int,
    ) -> List[dict]:
        query_text = _extract_query_text(match_expressions)
        if not query_text:
            return []

        results = []
        seen_uids = set()
        for uid in uid_list:
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            try:
                hits = self._mem0.search(query_text, user_id=uid, limit=top_n * 2) or []
            except Exception as e:
                logger.error("mem0 search failed for user %s: %s", uid, e)
                continue
            # Handle both dict-with-results and list formats
            if isinstance(hits, dict):
                hits = hits.get("results") or hits.get("memories") or []
            for h in hits:
                meta = h.get("metadata", {})
                if meta.get("memory_id") in memory_ids:
                    results.append(h)

        # apply condition filters
        if condition_dict:
            results = self._filter_by_condition(results, condition_dict)

        # sort by score (mem0 returns scored results) then limit
        results = results[:top_n]
        return [_metadata_to_msg(r) for r in results]

    def filter_by_keywords(
        self, keywords: List[str], uid: str, memory_ids: List[str], top_n: int,
    ) -> List[dict]:
        """
        Filter messages by keywords using in-memory filtering.
        Note: mem0 doesn't have native keyword filtering, so we fetch all and filter.
        """
        if not keywords:
            return []
        import json
        records = self._get_all_for_memories([uid], memory_ids)

        # Filter by keywords: match any keyword in the message's keywords list
        filtered = []
        for r in records:
            meta = r.get("metadata", {})
            msg_keywords_str = meta.get("keywords", "[]")
            try:
                msg_keywords = json.loads(msg_keywords_str) if isinstance(msg_keywords_str, str) else msg_keywords_str
            except (json.JSONDecodeError, TypeError):
                msg_keywords = []

            # Check if any of the filter keywords match
            if any(kw in msg_keywords for kw in keywords):
                filtered.append(r)

        # Sort by valid_at desc and limit
        filtered.sort(key=lambda r: r.get("metadata", {}).get("valid_at", ""), reverse=True)
        return [_metadata_to_msg(r) for r in filtered[:top_n]]

    # ── size / eviction ──

    def calculate_message_size(self, message: dict) -> int:
        content = message.get("content", "")
        size = sys.getsizeof(content)
        # Estimate embedding size: assume float32 (4 bytes per element)
        embed = message.get("content_embed")
        if embed and isinstance(embed, list) and len(embed) > 0:
            size += len(embed) * 4
        return size

    def calculate_memory_size(self, memory_ids: List[str], uid_list: List[str]) -> dict:
        records = self._get_all_for_memories(uid_list, memory_ids)
        size_dict: dict = {}
        for r in records:
            msg = _metadata_to_msg(r)
            mid = msg["memory_id"]
            # Note: mem0 manages embeddings internally; we only count content size
            # (unlike native_store which estimates embedding size)
            size_dict[mid] = size_dict.get(mid, 0) + sys.getsizeof(msg.get("content", ""))
        return size_dict

    def pick_messages_to_delete_by_fifo(
        self, memory_id: str, uid: str, size_to_delete: int,
    ) -> tuple[list, int]:
        records = self._get_all_for_memory(uid, memory_id)

        # prioritise forgotten messages (those with forget_at set)
        forgotten = [r for r in records if r.get("metadata", {}).get("forget_at")]
        forgotten.sort(key=lambda r: r.get("metadata", {}).get("forget_at", ""))

        active = [r for r in records if not r.get("metadata", {}).get("forget_at")]
        active.sort(key=lambda r: r.get("metadata", {}).get("valid_at", ""))

        ids_to_remove: list = []
        current_size = 0
        for r in forgotten + active:
            if current_size >= size_to_delete:
                break
            content = r.get("memory", "")
            current_size += sys.getsizeof(content)
            ids_to_remove.append(r.get("metadata", {}).get("message_id", 0))
        return ids_to_remove, current_size

    # ── misc ──

    def get_missing_field_messages(self, memory_id: str, uid: str, field_name: str) -> List[dict]:
        # mem0 doesn't have tokenized fields; always return empty
        return []

    def get_by_message_id(self, memory_id: str, message_id: int, uid: str) -> dict | None:
        records = self._get_all_for_memory(uid, memory_id)
        for r in records:
            if r.get("metadata", {}).get("message_id") == message_id:
                return _metadata_to_msg(r)
        return None

    def get_max_message_id(self, uid_list: List[str], memory_ids: List[str]) -> int:
        records = self._get_all_for_memories(uid_list, memory_ids)
        if not records:
            return 1
        max_id = max(
            (r.get("metadata", {}).get("message_id", 0) for r in records),
            default=1,
        )
        return max(max_id, 1)

    # ── internal filtering ──

    @staticmethod
    def _filter_by_condition(records: list[dict], condition: dict) -> list[dict]:
        """Filter mem0 records by a RAGFlow-style condition dict."""
        result = records
        for k, v in condition.items():
            if v is None or v == "":
                continue
            if isinstance(v, list):
                result = [r for r in result if r.get("metadata", {}).get(k) in v]
            else:
                result = [r for r in result if r.get("metadata", {}).get(k) == v]
        return result
