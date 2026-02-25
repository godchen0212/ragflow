"""Memory backend factory and exports."""

from memory.backends.base import MemoryBackend, MemoryRecord, AddMemoryRequest


def get_memory_backend(backend_type: str = None, config: dict = None) -> MemoryBackend:
    """Factory function to get the appropriate memory backend.

    Args:
        backend_type: "native" or "mem0". If None, uses MEMORY_BACKEND setting.
        config: Backend-specific configuration dict.

    Returns:
        MemoryBackend instance.
    """
    from common import settings

    _type = (backend_type or getattr(settings, "MEMORY_BACKEND", "native")).lower()

    if _type == "mem0":
        from memory.backends.mem0_backend import Mem0MemoryBackend
        return Mem0MemoryBackend(config=config)

    from memory.backends.native_backend import NativeMemoryBackend
    return NativeMemoryBackend()


__all__ = [
    "MemoryBackend",
    "MemoryRecord",
    "AddMemoryRequest",
    "get_memory_backend",
]
