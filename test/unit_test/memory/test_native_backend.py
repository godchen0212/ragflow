"""Unit tests for NativeMemoryBackend."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from memory.backends.native_backend import NativeMemoryBackend
from memory.backends.base import AddMemoryRequest, MemoryRecord


class TestNativeMemoryBackend:
    """Test suite for NativeMemoryBackend."""

    @pytest.fixture
    def backend(self):
        """Create a NativeMemoryBackend instance."""
        return NativeMemoryBackend()

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

    @pytest.mark.asyncio
    @patch("memory.backends.native_backend.MessageService.get_recent_messages")
    @patch("memory.backends.native_backend.save_to_memory")
    async def test_add_memory_success(self, mock_save, mock_get_recent, backend, sample_request):
        """Test successful memory addition."""
        mock_save.return_value = (True, "Success")
        mock_get_recent.return_value = [
            {
                "message_id": "msg-001",
                "message_type": "raw",
                "content": "Test content",
                "keywords": ["test"],
                "agent_id": "agent-001",
                "session_id": "session-001",
                "user_id": "user-001",
                "valid_at": "2025-01-01T00:00:00Z",
                "invalid_at": None,
                "status": True,
            }
        ]

        records = await backend.add_memory(sample_request)

        assert len(records) > 0
        assert records[0].memory_id == "mem-001"
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch("memory.backends.native_backend.save_to_memory")
    async def test_add_memory_failure(self, mock_save, backend, sample_request):
        """Test memory addition failure."""
        mock_save.return_value = (False, "Error")

        records = await backend.add_memory(sample_request)

        assert records == []
        mock_save.assert_called_once()

    @patch("memory.backends.native_backend.query_message")
    def test_search_memory(self, mock_query, backend):
        """Test memory search."""
        mock_query.return_value = [
            {
                "message_id": "msg-001",
                "message_type": "semantic",
                "content": "Python content",
                "keywords": ["python"],
                "agent_id": "agent-001",
                "session_id": "session-001",
                "user_id": "user-001",
                "valid_at": "2025-01-01T00:00:00Z",
                "invalid_at": None,
                "status": True,
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
        mock_query.assert_called_once()

    @patch("memory.backends.native_backend.MessageService.get_by_message_id")
    def test_get_memory_found(self, mock_get, backend):
        """Test getting a specific memory."""
        mock_get.return_value = {
            "message_id": "msg-001",
            "message_type": "raw",
            "content": "Test",
            "keywords": ["test"],
            "agent_id": "agent-001",
            "session_id": "session-001",
            "user_id": "user-001",
            "valid_at": "2025-01-01T00:00:00Z",
            "invalid_at": None,
            "status": True,
        }

        record = backend.get_memory("mem-001", "msg-001", "user-001")

        assert record is not None
        assert record.message_id == "msg-001"

    @patch("memory.backends.native_backend.MessageService.get_by_message_id")
    def test_get_memory_not_found(self, mock_get, backend):
        """Test getting a memory that doesn't exist."""
        mock_get.return_value = None

        record = backend.get_memory("mem-001", "msg-999", "user-001")

        assert record is None

    @patch("memory.backends.native_backend.MessageService.update_message")
    def test_update_memory(self, mock_update, backend):
        """Test updating a memory."""
        mock_update.return_value = True

        result = backend.update_memory(
            memory_id="mem-001",
            message_id="msg-001",
            user_id="user-001",
            content="Updated content",
        )

        assert result is True
        mock_update.assert_called_once()

    @patch("memory.backends.native_backend.MessageService.delete_message")
    def test_delete_memory(self, mock_delete, backend):
        """Test deleting a memory."""
        mock_delete.return_value = 1

        result = backend.delete_memory("mem-001", "msg-001", "user-001")

        assert result is True

    @patch("memory.backends.native_backend.MessageService.delete_message")
    def test_delete_memory_not_found(self, mock_delete, backend):
        """Test deleting a memory that doesn't exist."""
        mock_delete.return_value = 0

        result = backend.delete_memory("mem-001", "msg-999", "user-001")

        assert result is False

    @patch("memory.backends.native_backend.MessageService.list_message")
    def test_list_memories(self, mock_list, backend):
        """Test listing memories."""
        mock_list.return_value = {
            "message_list": [
                {
                    "message_id": "msg-001",
                    "message_type": "raw",
                    "content": "Test",
                    "keywords": ["test"],
                    "agent_id": "agent-001",
                    "session_id": "session-001",
                    "user_id": "user-001",
                    "valid_at": "2025-01-01T00:00:00Z",
                    "invalid_at": None,
                    "status": True,
                }
            ],
            "total_count": 1,
        }

        records, total = backend.list_memories("mem-001", "user-001")

        assert len(records) == 1
        assert total == 1

    @patch("memory.backends.native_backend.MessageService.filter_by_keywords")
    def test_filter_by_keywords(self, mock_filter, backend):
        """Test filtering by keywords."""
        mock_filter.return_value = [
            {
                "message_id": "msg-001",
                "message_type": "raw",
                "content": "Python content",
                "keywords": ["python"],
                "agent_id": "agent-001",
                "session_id": "session-001",
                "user_id": "user-001",
                "valid_at": "2025-01-01T00:00:00Z",
                "invalid_at": None,
                "status": True,
            }
        ]

        records = backend.filter_by_keywords("mem-001", "user-001", ["python"])

        assert len(records) == 1
        assert "python" in records[0].keywords
