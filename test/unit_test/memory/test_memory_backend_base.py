"""Unit tests for MemoryBackend base class."""

import pytest
from memory.backends.base import MemoryBackend, MemoryRecord, AddMemoryRequest


class TestMemoryRecord:
    """Test suite for MemoryRecord dataclass."""

    def test_memory_record_creation(self):
        """Test creating a MemoryRecord with all fields."""
        record = MemoryRecord(
            memory_id="mem-001",
            message_id="msg-001",
            content="Test content",
            message_type="semantic",
            keywords=["test", "content"],
            agent_id="agent-001",
            session_id="session-001",
            user_id="user-001",
            valid_at="2025-01-01T00:00:00Z",
            invalid_at=None,
            status=True,
        )
        assert record.memory_id == "mem-001"
        assert record.message_id == "msg-001"
        assert record.content == "Test content"
        assert record.message_type == "semantic"
        assert record.keywords == ["test", "content"]
        assert record.score == 0.0
        assert record.metadata == {}

    def test_memory_record_defaults(self):
        """Test MemoryRecord default values."""
        record = MemoryRecord(
            memory_id="mem-001",
            message_id="msg-001",
            content="Test",
            message_type="raw",
            keywords=[],
            agent_id="agent-001",
            session_id="session-001",
            user_id="user-001",
            valid_at="2025-01-01T00:00:00Z",
            invalid_at=None,
            status=True,
        )
        assert record.score == 0.0
        assert record.metadata == {}

    def test_memory_record_with_metadata(self):
        """Test MemoryRecord with custom metadata."""
        metadata = {"custom_field": "value"}
        record = MemoryRecord(
            memory_id="mem-001",
            message_id="msg-001",
            content="Test",
            message_type="episodic",
            keywords=[],
            agent_id="agent-001",
            session_id="session-001",
            user_id="user-001",
            valid_at="2025-01-01T00:00:00Z",
            invalid_at=None,
            status=True,
            metadata=metadata,
        )
        assert record.metadata == metadata


class TestAddMemoryRequest:
    """Test suite for AddMemoryRequest dataclass."""

    def test_add_memory_request_creation(self):
        """Test creating an AddMemoryRequest."""
        request = AddMemoryRequest(
            user_input="Hello",
            agent_response="Hi there",
            user_id="user-001",
            agent_id="agent-001",
            session_id="session-001",
            memory_id="mem-001",
        )
        assert request.user_input == "Hello"
        assert request.agent_response == "Hi there"
        assert request.user_id == "user-001"
        assert request.agent_id == "agent-001"
        assert request.session_id == "session-001"
        assert request.memory_id == "mem-001"
        assert request.metadata == {}

    def test_add_memory_request_with_metadata(self):
        """Test AddMemoryRequest with metadata."""
        metadata = {"source": "api"}
        request = AddMemoryRequest(
            user_input="Hello",
            agent_response="Hi",
            user_id="user-001",
            agent_id="agent-001",
            session_id="session-001",
            memory_id="mem-001",
            metadata=metadata,
        )
        assert request.metadata == metadata


class TestMemoryBackendAbstract:
    """Test suite for MemoryBackend abstract class."""

    def test_memory_backend_is_abstract(self):
        """Test that MemoryBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryBackend()

    def test_memory_backend_requires_all_methods(self):
        """Test that subclasses must implement all abstract methods."""

        class IncompleteBackend(MemoryBackend):
            async def add_memory(self, request):
                pass

        # Should raise TypeError because not all methods are implemented
        with pytest.raises(TypeError):
            IncompleteBackend()
