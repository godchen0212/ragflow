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
Native memory store implementation.

Delegates to the pluggable doc-store connection (ES / Infinity / OceanBase)
exposed via ``settings.msgStoreConn``.  This is a 1-to-1 extraction of the
original ``MessageService`` logic — no behavioural changes.
"""
from __future__ import annotations

import sys
from typing import List

from common.doc_store.doc_store_base import OrderByExpr, MatchExpr
from memory.services.base import MemoryStoreBase

_RAW_MESSAGE_TYPE = "raw"


class NativeMemoryStore(MemoryStoreBase):

    def __init__(self, msg_store_conn):
        self._conn = msg_store_conn

    @staticmethod
    def _index_name(uid: str) -> str:
        return f"memory_{uid}"

    # ── index management ──

    def has_index(self, uid: str, memory_id: str) -> bool:
        return self._conn.index_exist(self._index_name(uid), memory_id)

    def create_index(self, uid: str, memory_id: str, vector_size: int) -> bool:
        return self._conn.create_idx(self._index_name(uid), memory_id, vector_size)

    def delete_index(self, uid: str, memory_id: str) -> None:
        return self._conn.delete_idx(self._index_name(uid), memory_id)

    # ── CRUD ──

    def insert_message(self, messages: List[dict], uid: str, memory_id: str) -> List[str]:
        index = self._index_name(uid)
        for m in messages:
            m.update({
                "id": f'{memory_id}_{m["message_id"]}',
                "status": 1 if m["status"] else 0,
                "keywords": m.get("keywords", []),
            })
        return self._conn.insert(messages, index, memory_id)

    def update_message(self, condition: dict, update_dict: dict, uid: str, memory_id: str) -> bool:
        index = self._index_name(uid)
        if "status" in update_dict:
            update_dict["status"] = 1 if update_dict["status"] else 0
        return self._conn.update(condition, update_dict, index, memory_id)

    def delete_message(self, condition: dict, uid: str, memory_id: str) -> int:
        return self._conn.delete(condition, self._index_name(uid), memory_id)

    # ── query ──

    def list_message(
        self, uid: str, memory_id: str,
        agent_ids: List[str] | None = None,
        keywords: str | None = None,
        page: int = 1, page_size: int = 50,
    ) -> dict:
        index = self._index_name(uid)
        filter_dict: dict = {}
        if agent_ids:
            filter_dict["agent_id"] = agent_ids
        if keywords:
            filter_dict["session_id"] = keywords

        select_fields = [
            "message_id", "message_type", "source_id", "memory_id",
            "user_id", "agent_id", "session_id", "valid_at",
            "invalid_at", "forget_at", "status", "keywords",
        ]
        order_by = OrderByExpr()
        order_by.desc("valid_at")

        res, total_count = self._conn.search(
            select_fields=select_fields,
            highlight_fields=[],
            condition={**filter_dict, "message_type": _RAW_MESSAGE_TYPE},
            match_expressions=[], order_by=order_by,
            offset=(page - 1) * page_size, limit=page_size,
            index_names=index, memory_ids=[memory_id],
            agg_fields=[], hide_forgotten=False,
        )
        if not total_count:
            return {"message_list": [], "total_count": 0}

        raw_msg_mapping = self._conn.get_fields(res, select_fields)
        raw_messages = list(raw_msg_mapping.values())

        extract_filter = {"source_id": [r["message_id"] for r in raw_messages]}
        extract_res, _ = self._conn.search(
            select_fields=select_fields,
            highlight_fields=[],
            condition=extract_filter,
            match_expressions=[], order_by=order_by,
            offset=0, limit=512,
            index_names=index, memory_ids=[memory_id],
            agg_fields=[], hide_forgotten=False,
        )
        extract_msg = self._conn.get_fields(extract_res, select_fields)
        grouped_extract_msg: dict = {}
        for msg in extract_msg.values():
            grouped_extract_msg.setdefault(msg["source_id"], []).append(msg)

        for raw_msg in raw_messages:
            raw_msg["extract"] = grouped_extract_msg.get(raw_msg["message_id"], [])

        return {"message_list": raw_messages, "total_count": total_count}

    def get_recent_messages(
        self, uid_list: List[str], memory_ids: List[str],
        agent_id: str, session_id: str, limit: int,
    ) -> List[dict]:
        index_names = [self._index_name(uid) for uid in uid_list]
        condition_dict = {"agent_id": agent_id, "session_id": session_id}
        order_by = OrderByExpr()
        order_by.desc("valid_at")

        fields = [
            "message_id", "message_type", "source_id", "memory_id",
            "user_id", "agent_id", "session_id", "valid_at",
            "invalid_at", "forget_at", "status", "content", "keywords",
        ]
        res, total_count = self._conn.search(
            select_fields=fields,
            highlight_fields=[],
            condition=condition_dict,
            match_expressions=[], order_by=order_by,
            offset=0, limit=limit,
            index_names=index_names, memory_ids=memory_ids, agg_fields=[],
        )
        if not total_count:
            return []
        doc_mapping = self._conn.get_fields(res, fields)
        return list(doc_mapping.values())

    def search_message(
        self, memory_ids: List[str], condition_dict: dict,
        uid_list: List[str], match_expressions: list[MatchExpr], top_n: int,
    ) -> List[dict]:
        index_names = [self._index_name(uid) for uid in uid_list]
        if "status" not in condition_dict:
            condition_dict["status"] = 1

        order_by = OrderByExpr()
        order_by.desc("valid_at")

        fields = [
            "message_id", "message_type", "source_id", "memory_id",
            "user_id", "agent_id", "session_id", "valid_at",
            "invalid_at", "forget_at", "status", "content", "keywords",
        ]
        res, total_count = self._conn.search(
            select_fields=fields,
            highlight_fields=[],
            condition=condition_dict,
            match_expressions=match_expressions,
            order_by=order_by,
            offset=0, limit=top_n,
            index_names=index_names, memory_ids=memory_ids, agg_fields=[],
        )
        if not total_count:
            return []
        docs = self._conn.get_fields(res, fields)
        return list(docs.values())

    def filter_by_keywords(
        self, keywords: List[str], uid: str, memory_ids: List[str], top_n: int,
    ) -> List[dict]:
        if not keywords:
            return []
        index_name = self._index_name(uid)
        condition_dict = {"keywords": keywords}

        order_by = OrderByExpr()
        order_by.desc("valid_at")

        fields = [
            "message_id", "message_type", "source_id", "memory_id",
            "user_id", "agent_id", "session_id", "valid_at",
            "invalid_at", "forget_at", "status", "content", "keywords",
        ]
        res, total_count = self._conn.search(
            select_fields=fields,
            highlight_fields=[],
            condition=condition_dict,
            match_expressions=[],
            order_by=order_by,
            offset=0, limit=top_n,
            index_names=index_name, memory_ids=memory_ids, agg_fields=[],
        )
        if not total_count:
            return []
        docs = self._conn.get_fields(res, fields)
        return list(docs.values())

    # ── size / eviction ──

    def calculate_message_size(self, message: dict) -> int:
        content_size = sys.getsizeof(message.get("content", ""))
        # Estimate embedding size: assume float32 (4 bytes per element)
        embed = message.get("content_embed", [])
        embed_size = len(embed) * 4 if embed else 0
        return content_size + embed_size

    def calculate_memory_size(self, memory_ids: List[str], uid_list: List[str]) -> dict:
        index_names = [self._index_name(uid) for uid in uid_list]
        order_by = OrderByExpr()
        order_by.desc("valid_at")

        # Limit: 2048 per memory, but cap at 65536 to avoid excessive queries
        limit = min(2048 * len(memory_ids), 65536)
        res, count = self._conn.search(
            select_fields=["memory_id", "content", "content_embed"],
            highlight_fields=[],
            condition={},
            match_expressions=[],
            order_by=order_by,
            offset=0, limit=limit,
            index_names=index_names, memory_ids=memory_ids,
            agg_fields=[], hide_forgotten=False,
        )
        if count == 0:
            return {}

        docs = self._conn.get_fields(res, ["memory_id", "content", "content_embed"])
        size_dict: dict = {}
        for doc in docs.values():
            size_dict[doc["memory_id"]] = size_dict.get(doc["memory_id"], 0) + self.calculate_message_size(doc)
        return size_dict

    def pick_messages_to_delete_by_fifo(
        self, memory_id: str, uid: str, size_to_delete: int,
    ) -> tuple[list, int]:
        select_fields = ["message_id", "content", "content_embed"]
        _index = self._index_name(uid)

        res = self._conn.get_forgotten_messages(select_fields, _index, memory_id)
        current_size = 0
        ids_to_remove: list = []

        if res:
            message_list = self._conn.get_fields(res, select_fields)
            for message in message_list.values():
                if current_size < size_to_delete:
                    current_size += self.calculate_message_size(message)
                    ids_to_remove.append(message["message_id"])
                else:
                    return ids_to_remove, current_size
            if current_size >= size_to_delete:
                return ids_to_remove, current_size

        order_by = OrderByExpr()
        order_by.asc("valid_at")
        res, _ = self._conn.search(
            select_fields=select_fields,
            highlight_fields=[],
            condition={},
            match_expressions=[],
            order_by=order_by,
            offset=0, limit=512,
            index_names=[_index], memory_ids=[memory_id], agg_fields=[],
        )
        docs = self._conn.get_fields(res, select_fields)
        for doc in docs.values():
            if current_size < size_to_delete:
                current_size += self.calculate_message_size(doc)
                ids_to_remove.append(doc["message_id"])
            else:
                return ids_to_remove, current_size
        return ids_to_remove, current_size

    # ── misc ──

    def get_missing_field_messages(self, memory_id: str, uid: str, field_name: str) -> List[dict]:
        select_fields = ["message_id", "content"]
        _index = self._index_name(uid)
        res = self._conn.get_missing_field_message(
            select_fields=select_fields,
            index_name=_index,
            memory_id=memory_id,
            field_name=field_name,
        )
        if not res:
            return []
        docs = self._conn.get_fields(res, select_fields)
        return list(docs.values())

    def get_by_message_id(self, memory_id: str, message_id: int, uid: str) -> dict | None:
        doc_id = f'{memory_id}_{message_id}'
        return self._conn.get(doc_id, self._index_name(uid), [memory_id])

    def get_max_message_id(self, uid_list: List[str], memory_ids: List[str]) -> int:
        order_by = OrderByExpr()
        order_by.desc("message_id")
        index_names = [self._index_name(uid) for uid in uid_list]

        res, total_count = self._conn.search(
            select_fields=["message_id"],
            highlight_fields=[],
            condition={},
            match_expressions=[],
            order_by=order_by,
            offset=0, limit=1,
            index_names=index_names, memory_ids=memory_ids,
            agg_fields=[], hide_forgotten=False,
        )
        if not total_count:
            return 1
        docs = self._conn.get_fields(res, ["message_id"])
        if not docs:
            return 1
        latest_msg = list(docs.values())[0]
        return int(latest_msg["message_id"])
