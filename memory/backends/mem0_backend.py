"""mem0 memory backend adapter.

Integrates mem0ai as an alternative memory backend.
Note: mem0 manages its own vector DB and bypasses RAGFlow's native storage.
"""

from typing import Optional
from memory.backends.base import MemoryBackend, MemoryRecord, AddMemoryRequest
from memory.utils.keyword_extractor import extract_keywords


class Mem0MemoryBackend(MemoryBackend):
    """mem0 backend using mem0ai library for memory management."""

    def __init__(self, config: dict = None):
        """Initialize mem0 backend.

        Args:
            config: mem0 configuration dict. If None, uses defaults.
                   Expected keys: vector_store, llm, etc.
        """
        try:
            from mem0 import Memory as Mem0Memory
        except ImportError:
            raise ImportError(
                "mem0ai is not installed. Install it with: pip install mem0ai"
            )

        self._mem0 = Mem0Memory.from_config(config or {})

    async def add_memory(self, request: AddMemoryRequest) -> list[MemoryRecord]:
        """Add a new memory from a conversation turn.

        Uses mem0's built-in LLM extraction pipeline.
        Note: RAGFlow's extract_by_llm() is NOT called - mem0 handles extraction.
        """
        messages = [
            {"role": "user", "content": request.user_input},
            {"role": "assistant", "content": request.agent_response},
        ]

        # mem0 does its own LLM extraction
        result = self._mem0.add(
            messages,
            user_id=request.user_id,
            metadata={
                "memory_id": request.memory_id,
                "agent_id": request.agent_id,
                "session_id": request.session_id,
            },
        )

        records = []
        for item in result.get("results", []):
            record = self._item_to_record(item, request)
            records.append(record)

        return records

    def search_memory(
        self,
        query: str,
        memory_id: str,
        user_id: str,
        filters: Optional[dict] = None,
        top_n: int = 5,
        threshold: float = 0.2,
    ) -> list[MemoryRecord]:
        """Search memory using mem0's search API."""
        results = self._mem0.search(query, user_id=user_id, limit=top_n)

        records = []
        for item in results:
            # Filter by memory_id if needed
            if item.get("metadata", {}).get("memory_id") == memory_id:
                record = self._item_to_record(item, None)
                records.append(record)

        return records[:top_n]

    def get_memory(
        self, memory_id: str, message_id: str, user_id: str
    ) -> Optional[MemoryRecord]:
        """Get a specific memory by ID.

        mem0 doesn't have a direct get-by-id API, so we search and filter.
        """
        all_records, _ = self.list_memories(memory_id, user_id, page=1, page_size=1000)
        for record in all_records:
            if record.message_id == message_id:
                return record
        return None

    def update_memory(
        self,
        memory_id: str,
        message_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Update an existing memory.

        mem0 doesn't support direct updates - would need to delete and re-add.
        """
        # mem0 doesn't have a native update API
        # This is a limitation of the mem0 backend
        return False

    def delete_memory(self, memory_id: str, message_id: str, user_id: str) -> bool:
        """Delete a memory.

        mem0 doesn't have a direct delete-by-id API.
        """
        # mem0 doesn't have a native delete API
        # This is a limitation of the mem0 backend
        return False

    def list_memories(
        self,
        memory_id: str,
        user_id: str,
        filters: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[MemoryRecord], int]:
        """List memories with pagination.

        mem0 doesn't have a native list API, so we fetch all and filter.
        """
        # mem0 doesn't support pagination natively
        # This is a limitation - we'd need to fetch all and slice
        all_items = self._mem0.get_all(user_id=user_id)

        records = []
        for item in all_items:
            if item.get("metadata", {}).get("memory_id") == memory_id:
                record = self._item_to_record(item, None)
                records.append(record)

        # Simple pagination
        start = (page - 1) * page_size
        end = start + page_size
        return records[start:end], len(records)

    def filter_by_keywords(
        self, memory_id: str, user_id: str, keywords: list[str], top_n: int = 20
    ) -> list[MemoryRecord]:
        """Filter memories by keywords.

        mem0 has no native keyword filter - fetch all and filter in-memory.
        """
        all_records, _ = self.list_memories(
            memory_id, user_id, page=1, page_size=1000
        )

        kw_set = {kw.lower() for kw in keywords}
        filtered = []

        for record in all_records:
            # Check if any keyword matches record keywords or content
            if any(kw in record.keywords for kw in kw_set) or any(
                kw in record.content.lower() for kw in kw_set
            ):
                filtered.append(record)

        return filtered[:top_n]

    @staticmethod
    def _item_to_record(item: dict, request: Optional[AddMemoryRequest]) -> MemoryRecord:
        """Convert a mem0 item to a MemoryRecord."""
        content = item.get("memory", "")
        keywords = extract_keywords(content)

        return MemoryRecord(
            memory_id=item.get("metadata", {}).get("memory_id", ""),
            message_id=str(item.get("id", "")),
            content=content,
            message_type="semantic",  # mem0 doesn't distinguish types
            keywords=keywords,
            agent_id=item.get("metadata", {}).get("agent_id", ""),
            session_id=item.get("metadata", {}).get("session_id", ""),
            user_id=item.get("user_id", ""),
            valid_at=item.get("created_at"),
            invalid_at=None,
            status=True,
            score=item.get("score", 0.0),
            metadata=item.get("metadata", {}),
        )
