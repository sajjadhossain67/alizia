"""
ALIZIA AI - Memory Platform & Multi-Tier Retrieval
Implements PRD Sections 27, 28, 29, 30.
Supports 4 memory scopes: Session, User, Workspace, Agent.
Ranking formula: semantic_similarity * importance * recency * confidence.
"""

from __future__ import annotations
import math
import time
from typing import List, Optional
from packages.schemas.models import MemoryObject


class MemoryEngine:
    def __init__(self):
        # In-memory store for MVP / development, backed by Postgres/pgvector in production
        self._store: List[MemoryObject] = []

    def add_memory(
        self,
        owner_id: str,
        scope: str,
        content: str,
        importance: float = 0.5,
        confidence: float = 0.9,
        embedding: Optional[List[float]] = None
    ) -> MemoryObject:
        mem = MemoryObject(
            owner_id=owner_id,
            scope=scope, # session, user, workspace, agent
            content=content,
            embedding=embedding,
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
            created_at=time.time(),
            last_used_at=time.time()
        )
        self._store.append(mem)
        return mem

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Lightweight lexical-semantic overlap for memory ranking"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.1
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        return len(intersection) / len(union) if union else 0.0

    def retrieve_relevant_memories(
        self,
        owner_id: str,
        query: str,
        scope: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.05
    ) -> List[MemoryObject]:
        """
        Section 29: Memory Retrieval Ranking
        Rank = semantic_similarity * importance * recency * confidence
        Never inject irrelevant memory.
        """
        now = time.time()
        candidates = []

        for mem in self._store:
            if mem.owner_id != owner_id:
                continue
            if scope and mem.scope != scope:
                continue

            similarity = self.compute_similarity(query, mem.content)
            # Recency factor: exponential decay over days
            age_hours = (now - mem.created_at) / 3600.0
            recency = math.exp(-0.01 * age_hours) # decay factor

            rank_score = similarity * mem.importance * recency * mem.confidence

            if rank_score >= threshold or similarity > 0.3:
                candidates.append((rank_score, mem))

        # Sort descending by rank score
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [c[1] for c in candidates[:top_k]]

        # Update last_used_at timestamp
        for r in results:
            r.last_used_at = now

        return results

    def delete_memory(self, memory_id: str, owner_id: str) -> bool:
        """Section 30: Memory Privacy (delete memory)"""
        initial_len = len(self._store)
        self._store = [m for m in self._store if not (m.id == memory_id and m.owner_id == owner_id)]
        return len(self._store) < initial_len

    def export_memories(self, owner_id: str) -> List[MemoryObject]:
        """Section 30: Memory Privacy (export memories)"""
        return [m for m in self._store if m.owner_id == owner_id]
