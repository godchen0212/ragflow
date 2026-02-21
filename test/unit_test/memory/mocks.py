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
In-memory mocks for unit-testing memory store implementations.

* ``MockMsgStoreConn``  — drop-in replacement for ``memory.utils.es_conn.ESConnection``
* ``MockMem0Memory``    — drop-in replacement for ``mem0.Memory``
"""
from __future__ import annotations

import copy
import re
from typing import Any


# ---------------------------------------------------------------------------
# MockMsgStoreConn — used by NativeMemoryStore tests
# ---------------------------------------------------------------------------

class MockMsgStoreConn:
    """
    Minimal in-memory implementation of the doc-store connection interface
    used by ``NativeMemoryStore``.  Stores documents in nested dicts keyed
    by ``(index_name, doc_id)``.
    """

    def __init__(self):
        # {index_name: {doc_id: doc_dict}}
        self._indices: dict[str, dict[str, dict]] = {}
        # track which (index, memory_id) pairs have been "created"
        self._created: set[tuple[str, str]] = set()

    # ── index ops ──

    def index_exist(self, index_name: str, memory_id: str | None = None) -> bool:
        if memory_id:
            return (index_name, memory_id) in self._created
        return index_name in self._indices

    def create_idx(self, index_name: str, memory_id: str, vector_size: int, parser_id: str | None = None) -> bool:
        self._indices.setdefault(index_name, {})
        self._created.add((index_name, memory_id))
        return True

    def delete_idx(self, index_name: str, memory_id: str) -> None:
        if index_name in self._indices:
            to_del = [k for k, v in self._indices[index_name].items()
                      if v.get("memory_id") == memory_id]
            for k in to_del:
                del self._indices[index_name][k]
        self._created.discard((index_name, memory_id))

    # ── CRUD ──

    def insert(self, rows: list[dict], index_name: str, memory_id: str | None = None) -> list[str]:
        store = self._indices.setdefault(index_name, {})
        errors: list[str] = []
        for row in rows:
            doc = copy.deepcopy(row)
            doc_id = doc.pop("id", None)
            if doc_id is None:
                errors.append("missing id")
                continue
            if memory_id:
                doc["memory_id"] = memory_id
            store[doc_id] = doc
        return errors

    def get(self, doc_id: str, index_name: str, dataset_ids: list[str] | None = None) -> dict | None:
        store = self._indices.get(index_name, {})
        doc = store.get(doc_id)
        if doc is None:
            return None
        result = copy.deepcopy(doc)
        result["id"] = doc_id
        return result

    def update(self, condition: dict, new_value: dict, index_name: str, memory_id: str) -> bool:
        store = self._indices.get(index_name, {})
        matched = self._match(store, condition, memory_id)
        if not matched:
            return False
        for doc_id in matched:
            for k, v in new_value.items():
                store[doc_id][k] = v
        return True

    def delete(self, condition: dict, index_name: str, memory_id: str) -> int:
        store = self._indices.get(index_name, {})
        matched = self._match(store, condition, memory_id)
        for doc_id in matched:
            del store[doc_id]
        return len(matched)

    def search(
        self,
        select_fields: list[str],
        highlight_fields: list[str],
        condition: dict,
        match_expressions: list,
        order_by: Any,
        offset: int,
        limit: int,
        index_names: str | list[str],
        memory_ids: list[str],
        agg_fields: list[str] | None = None,
        rank_feature: dict | None = None,
        hide_forgotten: bool = True,
    ):
        if isinstance(index_names, str):
            index_names = [index_names]

        all_docs: list[tuple[str, dict]] = []
        for idx in index_names:
            store = self._indices.get(idx, {})
            for doc_id, doc in store.items():
                if doc.get("memory_id") not in memory_ids:
                    continue
                if hide_forgotten and doc.get("forget_at"):
                    continue
                if not self._doc_matches_condition(doc, condition):
                    continue
                all_docs.append((doc_id, doc))

        # sort
        if order_by and hasattr(order_by, "fields") and order_by.fields:
            for field, direction in reversed(order_by.fields):
                reverse = direction == 1
                all_docs.sort(
                    key=lambda x: x[1].get(field, ""),
                    reverse=reverse,
                )

        total = len(all_docs)
        page = all_docs[offset:offset + limit] if limit > 0 else all_docs

        # build ES-like response
        hits = []
        for doc_id, doc in page:
            hit = {"_id": doc_id, "_source": {**doc, "id": doc_id}}
            hits.append(hit)

        res = {"hits": {"total": {"value": total}, "hits": hits}}
        return res, total

    def get_total(self, res) -> int:
        if not res:
            return 0
        return res.get("hits", {}).get("total", {}).get("value", 0)

    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        if not res:
            return {}
        result: dict[str, dict] = {}
        for hit in res.get("hits", {}).get("hits", []):
            src = hit["_source"]
            doc_id = hit["_id"]
            entry: dict = {}
            for f in fields:
                if f in src:
                    entry[f] = src[f]
            if entry:
                result[doc_id] = entry
        return result

    def get_forgotten_messages(self, select_fields: list[str], index_name: str, memory_id: str, limit: int = 512):
        store = self._indices.get(index_name, {})
        hits = []
        for doc_id, doc in store.items():
            if doc.get("memory_id") == memory_id and doc.get("forget_at"):
                hits.append({"_id": doc_id, "_source": {**doc, "id": doc_id}})
        hits.sort(key=lambda h: h["_source"].get("forget_at", ""))
        hits = hits[:limit]
        if not hits:
            return None
        return {"hits": {"total": {"value": len(hits)}, "hits": hits}}

    def get_missing_field_message(self, select_fields: list[str], index_name: str, memory_id: str, field_name: str, limit: int = 512):
        store = self._indices.get(index_name, {})
        hits = []
        for doc_id, doc in store.items():
            if doc.get("memory_id") == memory_id and field_name not in doc:
                hits.append({"_id": doc_id, "_source": {**doc, "id": doc_id}})
        if not hits:
            return None
        return {"hits": {"total": {"value": len(hits)}, "hits": hits[:limit]}}

    # ── helpers ──

    @staticmethod
    def _doc_matches_condition(doc: dict, condition: dict) -> bool:
        for k, v in condition.items():
            if v is None or v == "":
                continue
            doc_val = doc.get(k)
            if isinstance(v, list):
                # Special handling for keywords: check if any filter keyword is in doc keywords
                if k == "keywords" and isinstance(doc_val, list):
                    if not any(kw in doc_val for kw in v):
                        return False
                else:
                    # For other list conditions: doc_val should be in the filter list
                    if doc_val not in v:
                        return False
            else:
                if doc_val != v:
                    return False
        return True

    @staticmethod
    def _match(store: dict[str, dict], condition: dict, memory_id: str) -> list[str]:
        matched = []
        for doc_id, doc in store.items():
            if doc.get("memory_id") != memory_id:
                continue
            ok = True
            for k, v in condition.items():
                if v is None:
                    continue
                dv = doc.get(k)
                if isinstance(v, list):
                    if dv not in v:
                        ok = False
                        break
                elif dv != v:
                    ok = False
                    break
            if ok:
                matched.append(doc_id)
        return matched


# ---------------------------------------------------------------------------
# MockMem0Memory — used by Mem0MemoryStore tests
# ---------------------------------------------------------------------------

class MockMem0Memory:
    """
    Minimal in-memory mock of ``mem0.Memory``.
    Implements ``add``, ``search``, ``get_all``, ``get``, ``update``,
    ``delete``, ``delete_all``.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}  # {mem0_id: record}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"mem0_{self._counter}"

    def add(self, content: str, user_id: str = "", agent_id: str = "",
            metadata: dict | None = None, **kwargs) -> dict:
        mid = self._next_id()
        record = {
            "id": mid,
            "memory": content,
            "metadata": {**(metadata or {}), "user_id_scope": user_id},
            "user_id": user_id,
        }
        self._store[mid] = record
        return {"id": mid}

    def search(self, query: str, user_id: str = "", limit: int = 10, **kwargs) -> list[dict]:
        results = []
        for r in self._store.values():
            if r.get("user_id") != user_id:
                continue
            # simple substring match for testing
            content = r.get("memory", "")
            if query.lower() in content.lower() or True:  # return all for simplicity
                results.append(copy.deepcopy(r))
        return results[:limit]

    def get_all(self, user_id: str = "", **kwargs) -> list[dict]:
        return [copy.deepcopy(r) for r in self._store.values()
                if r.get("user_id") == user_id]

    def get(self, memory_id: str, **kwargs) -> dict | None:
        r = self._store.get(memory_id)
        return copy.deepcopy(r) if r else None

    def update(self, memory_id: str, data: str = "", metadata: dict | None = None, **kwargs) -> dict:
        r = self._store.get(memory_id)
        if r is None:
            raise ValueError(f"Memory {memory_id} not found")
        if data:
            r["memory"] = data
        if metadata is not None:
            r["metadata"] = metadata
        return copy.deepcopy(r)

    def delete(self, memory_id: str, **kwargs) -> None:
        self._store.pop(memory_id, None)

    def delete_all(self, user_id: str = "", **kwargs) -> None:
        to_del = [k for k, v in self._store.items() if v.get("user_id") == user_id]
        for k in to_del:
            del self._store[k]

    @classmethod
    def from_config(cls, config: dict) -> "MockMem0Memory":
        return cls()
