import sys
sys.path.insert(0, r'C:\Users\sajja\Desktop\Tree\alizia ai')
sys.path.insert(0, r'C:\Users\sajja\Desktop\Tree\alizia ai\ai')
from ai.memory_engine.memory import MemoryEngine
from packages.schemas.models import MemoryObject

mem_engine = MemoryEngine()

# Add a memory
mem = mem_engine.add_memory(
    owner_id="user_123",
    scope="user",
    content="User prefers Python over JavaScript",
    importance=0.8,
    confidence=0.9
)
print(f"Added memory: {mem.id}")
print(f"Scope: {mem.scope}, Importance: {mem.importance}, Confidence: {mem.confidence}")

# Retrieve relevant memories
results = mem_engine.retrieve_relevant_memories(
    owner_id="user_123",
    query="Python programming",
    top_k=3
)
print(f"\nRetrieved {len(results)} memories:")
for m in results:
    print(f"  - {m.content[:50]}... importance={m.importance:.2f}, confidence={m.confidence:.2f}")

# Delete a memory
deleted = mem_engine.delete_memory(mem.id, "user_123")
print(f"\nDeleted memory: {deleted}")

# Export all memories
all_mems = mem_engine.export_memories("user_123")
print(f"\nExported {len(all_mems)} memories for user_123")