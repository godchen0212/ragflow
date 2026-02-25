"""Native RAGFlow memory backend adapter.

Wraps existing MessageService and memory_message_service to provide
a unified MemoryBackend interface.
"""

from typing import Optional
from memory.backends.base import MemoryBackend, MemoryRecord, AddMemoryRequest
from memory.services.messages import MessageService
from api.db.joint_services.memory_message_service import save_to_memory, query_message


class NativeMemoryBackend(MemoryBackend):
    """Native backend using RAGFlow's existing ES/Infinity/OceanBase storage."""

    async def add_memory(self, request: AddMemoryRequest) -> list[MemoryRecord]:
        """Add a new memory from a conversation turn.

        Uses RAGFlow's existing LLM extraction pipeline and storage.
        """
        success, msg = await save_to_memory(request.memory_id, {
            "user_id": request.user_id,
            "agent_id": request.agent_id,
            "session_id": request.session_id,
            "user_input": request.user_input,
            "agent_response": request.agent_response,
        })

        if not success:
            return []

        # Return recent records for this session
        recent = MessageService.get_recent_messages(
            uid_list=[request.user_id],
            memory_ids=[request.memory_id],
            agent_id=request.agent_id,
            session_id=request.session_id,
            limit=10
        )
        return [self._doc_to_record(doc, request.memory_id) for doc in recent]

    def search_memory(
        self,
        query: str,
        memory_id: str,
        user_id: str,
        filters: Optional[dict] = None,
        top_n: int = 5,
        threshold: float = 0.2,
    ) -> list[MemoryRecord]:
        """Search memory using hybrid search (text + vector)."""
        filter_dict = {"memory_id": [memory_id]}
        if filters:
            filter_dict.update(filters)

        params = {
            "query": query,
            "similarity_threshold": threshold,
            "keywords_similarity_weight": 0.7,
            "top_n": top_n,
        }

        docs = query_message(filter_dict, params)
        return [self._doc_to_record(doc, memory_id) for doc in docs]

    def get_memory(
        self, memory_id: str, message_id: str, user_id: str
    ) -> Optional[MemoryRecord]:
        """Get a specific memory by ID."""
        doc = MessageService.get_by_message_id(memory_id, int(message_id), user_id)
        if not doc:
            return None
        return self._doc_to_record(doc, memory_id)

    def update_memory(
        self,
        memory_id: str,
        message_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Update an existing memory."""
        update_dict = {}
        if content:
            update_dict["content"] = content
        if metadata:
            update_dict.update(metadata)

        if not update_dict:
            return True

        return MessageService.update_message(
            condition={"message_id": int(message_id)},
            update_dict=update_dict,
            uid=user_id,
            memory_id=memory_id,
        )

    def delete_memory(self, memory_id: str, message_id: str, user_id: str) -> bool:
        """Delete a memory."""
        deleted = MessageService.delete_message(
            condition={"message_id": int(message_id)},
            uid=user_id,
            memory_id=memory_id,
        )
        return deleted > 0

    def list_memories(
        self,
        memory_id: str,
        user_id: str,
        filters: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[MemoryRecord], int]:
        """List memories with pagination."""
        result = MessageService.list_message(
            uid=user_id,
            memory_id=memory_id,
            agent_ids=filters.get("agent_ids") if filters else None,
            keywords=filters.get("keywords") if filters else None,
            page=page,
            page_size=page_size,
        )

        messages = result.get("message_list", [])
        total = result.get("total_count", 0)
        records = [self._doc_to_record(msg, memory_id) for msg in messages]
        return records, total

    def filter_by_keywords(
        self, memory_id: str, user_id: str, keywords: list[str], top_n: int = 20
    ) -> list[MemoryRecord]:
        """Filter memories by keywords."""
        docs = MessageService.filter_by_keywords(user_id, memory_id, keywords, top_n)
        return [self._doc_to_record(doc, memory_id) for doc in docs]

    @staticmethod
    def _doc_to_record(doc: dict, memory_id: str) -> MemoryRecord:
        """Convert a message document to a MemoryRecord."""
        return MemoryRecord(
            memory_id=memory_id,
            message_id=str(doc.get("message_id", "")),
            content=doc.get("content", ""),
            message_type=doc.get("message_type", "raw"),
            keywords=doc.get("keywords", []),
            agent_id=doc.get("agent_id", ""),
            session_id=doc.get("session_id", ""),
            user_id=doc.get("user_id", ""),
            valid_at=doc.get("valid_at"),
            invalid_at=doc.get("invalid_at"),
            status=bool(doc.get("status", True)),
            score=doc.get("_score", 0.0),
            metadata=doc.get("metadata", {}),
        )
