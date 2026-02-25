#!/usr/bin/env python3
"""Verification script for memory backend functionality.

Usage:
    python scripts/verify_memory.py --memory-id <id> --user-id <uid>
    MEMORY_BACKEND=mem0 python scripts/verify_memory.py --memory-id <id> --user-id <uid>
"""

import argparse
import asyncio
import sys
from memory.backends import get_memory_backend, AddMemoryRequest


async def verify_memory_backend(memory_id: str, user_id: str):
    """Verify memory backend functionality."""
    print("=" * 60)
    print("RAGFlow Memory Backend Verification")
    print("=" * 60)

    # Get backend
    backend = get_memory_backend()
    backend_type = type(backend).__name__
    print(f"\n✓ Backend initialized: {backend_type}")

    # Test data
    test_request = AddMemoryRequest(
        user_input="I love Python and FastAPI for building APIs",
        agent_response="Great choice! FastAPI is excellent for async APIs.",
        user_id=user_id,
        agent_id="agent-test-001",
        session_id="session-test-001",
        memory_id=memory_id,
    )

    # Test 1: Add memory
    print("\n[1] Testing add_memory()...")
    try:
        records = await backend.add_memory(test_request)
        if records:
            print(f"✓ Added {len(records)} memory record(s)")
            for i, record in enumerate(records):
                print(f"  - Record {i+1}: {record.message_id}")
                print(f"    Content: {record.content[:50]}...")
                print(f"    Keywords: {record.keywords}")
        else:
            print("⚠ No records returned (may be expected for some backends)")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 2: Search memory
    print("\n[2] Testing search_memory()...")
    try:
        search_results = backend.search_memory(
            query="Python web framework",
            memory_id=memory_id,
            user_id=user_id,
            top_n=5,
        )
        if search_results:
            print(f"✓ Found {len(search_results)} result(s)")
            for i, record in enumerate(search_results):
                print(f"  - Result {i+1}: {record.content[:50]}...")
        else:
            print("⚠ No results found")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 3: Filter by keywords
    print("\n[3] Testing filter_by_keywords()...")
    try:
        keyword_results = backend.filter_by_keywords(
            memory_id=memory_id,
            user_id=user_id,
            keywords=["python", "fastapi"],
            top_n=20,
        )
        if keyword_results:
            print(f"✓ Found {len(keyword_results)} result(s) matching keywords")
            for i, record in enumerate(keyword_results):
                print(f"  - Result {i+1}: {record.content[:50]}...")
                print(f"    Keywords: {record.keywords}")
        else:
            print("⚠ No results found")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 4: List memories
    print("\n[4] Testing list_memories()...")
    try:
        memories, total = backend.list_memories(
            memory_id=memory_id,
            user_id=user_id,
            page=1,
            page_size=10,
        )
        print(f"✓ Listed {len(memories)} memory record(s) (total: {total})")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 5: Get specific memory
    if records:
        print("\n[5] Testing get_memory()...")
        try:
            record = backend.get_memory(
                memory_id=memory_id,
                message_id=records[0].message_id,
                user_id=user_id,
            )
            if record:
                print(f"✓ Retrieved memory: {record.message_id}")
                print(f"  Content: {record.content[:50]}...")
            else:
                print("⚠ Memory not found")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    print("\n" + "=" * 60)
    print("✓ Memory verification passed")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify RAGFlow memory backend functionality"
    )
    parser.add_argument(
        "--memory-id",
        required=True,
        help="Memory configuration ID",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User/tenant ID",
    )

    args = parser.parse_args()

    try:
        success = asyncio.run(verify_memory_backend(args.memory_id, args.user_id))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
