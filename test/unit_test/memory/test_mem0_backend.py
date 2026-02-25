"""Unit tests for Mem0MemoryBackend."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from memory.backends.mem0_backend import Mem0MemoryBackend
from memory.backends.base import AddMemoryRequest


class TestMem0MemoryBackend:
    """Test suite for Mem0MemoryBackend."""

    @pytest.fixture
    def mock_mem0(self):
        """Create a mock mem0 Memory instance."""
        return MagicMock()

    @pytest.fixture
    def backend(self, mock_mem0):
        """Create a Mem0MemoryBackend instance with mocked mem0."""
        with patch("memory.backends.mem0_backend.Mem0Memory") as mock_mem0_class:
            mock_mem0_class.from_config.return_value = mock_mem0
            return Mem0MemoryBackend(config={})

    @pytest.fixture
    def sample_request(self):
        """Create a sample AddMemoryRequest."""
        return AddMemoryRequest(
            user_input="I love Python",
            agent_response="Python is great!",
            user_id="user-001",
            agent_id="agent-001",
            session_id="session-001",
            memory_id="mem-001",
        )

    def test_mem0_import_error(self):
        """Test that ImportError is raised when mem0ai is not installed."""
        with patch("memory.backends.mem0_backend.Mem0Memory", side_effect=ImportError):
            with pytest.raises(ImportError, match="mem0ai is not installed"):
                Mem0MemoryBackend()

    @pytest.mark.asyncio
    async def test_add_memory_returns_records(self, backend, mock_mem0, sample_request):
        """Test successful memory addition returns records."""
        mock_mem0.add.return_value = {
            "results": [
                {
                    "id": "mem-item-001",
                    "memory": "Python is a great programming language",
                    "user_id": "user-001",
                    "created_at": "2025-01-01T00:00:00Z",
                    "metadata": {
                        "memory_id": "mem-001",
                        "agent_id": "agent-001",
                        "session_id": "session-001",
                    },
                }
            ]
        }

        records = await backend.add_memory(sample_request)

        assert len(records) == 1
        assert records[0].memory_id == "mem-001"
        assert records[0].content == "Python is a great programming language"
        mock_mem0.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memory_empty_results(self, backend, mock_mem0, sample_request):
        """Test memory addition with empty results."""
        mock_mem0.add.return_value = {"results": []}

        records = await backend.add_memory(sample_request)

        assert records == []

    def test_search_memory(self, backend, mock_mem0):
        """Test memory search."""
        mock_mem0.search.return_value = [
            {
                "id": "mem-item-001",
                "memory": "Python content",
                "user_id": "user-001",
                "created_at": "2025-01-01T00:00:00Z",
                "metadata": {"memory_id": "mem-001"},
            }
        ]

        records = backend.search_memory(
            query="Python",
            memory_id="mem-001",
            user_id="user-001",
            top_n=5,
        )

        assert len(records) == 1
        assert records[0].content == "Python content"

    def test_search_memory_filters_by_memory_id(self, backend, mock_mem0):
        """Test that search filters by memory_id."""
        mock_mem0.search.return_value = [
            {
                "id": "mem-item-001",
                "memory": "Content 1",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            },
            {
                "id": "mem-item-002",
                "memory": "Content 2",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-002"},
            },
        ]

        records = backend.search_memory(
            query="test",
            memory_id="mem-001",
            user_id="user-001",
        )

        # Should only return records for mem-001
        assert len(records) == 1
        assert records[0].memory_id == "mem-001"

    def test_get_memory_found(self, backend, mock_mem0):
        """Test getting a specific memory."""
        mock_mem0.get_all.return_value = [
            {
                "id": "mem-item-001",
                "memory": "Test content",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            }
        ]

        record = backend.get_memory("mem-001", "mem-item-001", "user-001")

        assert record is not None
        assert record.message_id == "mem-item-001"

    def test_get_memory_not_found(self, backend, mock_mem0):
        """Test getting a memory that doesn't exist."""
        mock_mem0.get_all.return_value = []

        record = backend.get_memory("mem-001", "mem-item-999", "user-001")

        assert record is None

    def test_update_memory_not_supported(self, backend):
        """Test that update is not supported in mem0 backend."""
        result = backend.update_memory(
            memory_id="mem-001",
            message_id="mem-item-001",
            user_id="user-001",
            content="Updated",
        )

        assert result is False

    def test_delete_memory_not_supported(self, backend):
        """Test that delete is not supported in mem0 backend."""
        result = backend.delete_memory("mem-001", "mem-item-001", "user-001")

        assert result is False

    def test_list_memories(self, backend, mock_mem0):
        """Test listing memories."""
        mock_mem0.get_all.return_value = [
            {
                "id": "mem-item-001",
                "memory": "Content 1",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            },
            {
                "id": "mem-item-002",
                "memory": "Content 2",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            },
        ]

        records, total = backend.list_memories("mem-001", "user-001", page=1, page_size=10)

        assert len(records) == 2
        assert total == 2

    def test_list_memories_pagination(self, backend, mock_mem0):
        """Test memory listing with pagination."""
        mock_mem0.get_all.return_value = [
            {"id": f"mem-item-{i:03d}", "memory": f"Content {i}", "user_id": "user-001", "metadata": {"memory_id": "mem-001"}}
            for i in range(1, 26)
        ]

        records, total = backend.list_memories("mem-001", "user-001", page=2, page_size=10)

        assert len(records) == 10
        assert total == 25

    def test_filter_by_keywords(self, backend, mock_mem0):
        """Test filtering by keywords."""
        mock_mem0.get_all.return_value = [
            {
                "id": "mem-item-001",
                "memory": "Python and FastAPI",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            },
            {
                "id": "mem-item-002",
                "memory": "JavaScript content",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            },
        ]

        records = backend.filter_by_keywords("mem-001", "user-001", ["python"])

        assert len(records) == 1
        assert "python" in records[0].content.lower()

    def test_filter_by_keywords_multiple(self, backend, mock_mem0):
        """Test filtering by multiple keywords."""
        mock_mem0.get_all.return_value = [
            {
                "id": "mem-item-001",
                "memory": "Python and FastAPI",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            },
            {
                "id": "mem-item-002",
                "memory": "JavaScript content",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            },
        ]

        records = backend.filter_by_keywords("mem-001", "user-001", ["python", "javascript"])

        assert len(records) == 2

    def test_filter_by_keywords_respects_top_n(self, backend, mock_mem0):
        """Test that filter_by_keywords respects top_n limit."""
        mock_mem0.get_all.return_value = [
            {
                "id": f"mem-item-{i:03d}",
                "memory": f"Python content {i}",
                "user_id": "user-001",
                "metadata": {"memory_id": "mem-001"},
            }
            for i in range(1, 26)
        ]

        records = backend.filter_by_keywords("mem-001", "user-001", ["python"], top_n=5)

        assert len(records) <= 5
