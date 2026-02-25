"""Abstract base class and data models for memory backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryRecord:
    """Represents a single memory record returned from backend operations."""

    memory_id: str
    message_id: str
    content: str
    message_type: str  # "raw", "semantic", "episodic", "procedural"
    keywords: list[str]  # extracted keywords
    agent_id: str
    session_id: str
    user_id: str
    valid_at: Optional[str]
    invalid_at: Optional[str]
    status: bool
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AddMemoryRequest:
    """Request to add a new memory from a conversation turn."""

    user_input: str
    agent_response: str
    user_id: str
    agent_id: str
    session_id: str
    memory_id: str
    metadata: dict = field(default_factory=dict)


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends.

    Implementations must support both native RAGFlow storage (ES/Infinity/OceanBase)
    and external backends like mem0.
    """

    @abstractmethod
    async def add_memory(self, request: AddMemoryRequest) -> list[MemoryRecord]:
        """Add a new memory from a conversation turn.

        Args:
            request: AddMemoryRequest with user input, agent response, and metadata.

        Returns:
            List of MemoryRecord objects created from the input.
        """
        pass

    @abstractmethod
    def search_memory(
        self,
        query: str,
        memory_id: str,
        user_id: str,
        filters: Optional[dict] = None,
        top_n: int = 5,
        threshold: float = 0.2,
    ) -> list[MemoryRecord]:
        """Search memory using hybrid search (text + vector).

        Args:
            query: Text query to search for.
            memory_id: Memory configuration ID.
            user_id: User ID for isolation.
            filters: Optional filter dict (e.g., {"message_type": "semantic"}).
            top_n: Number of results to return.
            threshold: Similarity threshold for vector search.

        Returns:
            List of MemoryRecord objects matching the query.
        """
        pass

    @abstractmethod
    def get_memory(
        self, memory_id: str, message_id: str, user_id: str
    ) -> Optional[MemoryRecord]:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory configuration ID.
            message_id: Message ID to retrieve.
            user_id: User ID for isolation.

        Returns:
            MemoryRecord if found, None otherwise.
        """
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        message_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: Memory configuration ID.
            message_id: Message ID to update.
            user_id: User ID for isolation.
            content: New content (optional).
            metadata: New metadata (optional).

        Returns:
            True if update succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str, message_id: str, user_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory configuration ID.
            message_id: Message ID to delete.
            user_id: User ID for isolation.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def list_memories(
        self,
        memory_id: str,
        user_id: str,
        filters: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[MemoryRecord], int]:
        """List memories with pagination.

        Args:
            memory_id: Memory configuration ID.
            user_id: User ID for isolation.
            filters: Optional filter dict.
            page: Page number (1-indexed).
            page_size: Number of results per page.

        Returns:
            Tuple of (list of MemoryRecord objects, total count).
        """
        pass

    @abstractmethod
    def filter_by_keywords(
        self, memory_id: str, user_id: str, keywords: list[str], top_n: int = 20
    ) -> list[MemoryRecord]:
        """Filter memories by keywords.

        Args:
            memory_id: Memory configuration ID.
            user_id: User ID for isolation.
            keywords: List of keywords to filter by (OR logic).
            top_n: Maximum number of results to return.

        Returns:
            List of MemoryRecord objects matching any of the keywords.
        """
        pass
