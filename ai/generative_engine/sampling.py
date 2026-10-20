"""
ALIZIA AI - Generative Sampling & Logits Processor
Implements Temperature scaling, Top-K, Nucleus (Top-P) sampling,
Repetition penalty, and Stop token detection.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class SamplingParams:
    """Parameters governing token generation and stochasticity."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 2048
    stop_sequences: List[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>", "</s>"])
    stop_token_ids: Set[int] = field(default_factory=lambda: {2, 5})
    seed: Optional[int] = None


class Sampler:
    """
    Applies logits transformations and samples the next token index.
    """

    @staticmethod
    def sample_next_token(
        logits: np.ndarray,
        params: SamplingParams,
        generated_token_ids: Optional[List[int]] = None,
        random_state: Optional[np.random.RandomState] = None
    ) -> int:
        """
        Processes raw unnormalized logits according to sampling parameters and returns next token ID.
        """
        logits = np.copy(logits)
        rng = random_state or (np.random.RandomState(params.seed) if params.seed is not None else np.random)

        # 1. Apply repetition penalty
        if generated_token_ids and params.repetition_penalty != 1.0:
            counts = {}
            for tid in generated_token_ids:
                counts[tid] = counts.get(tid, 0) + 1

            for tid, count in counts.items():
                if 0 <= tid < len(logits):
                    if logits[tid] > 0:
                        logits[tid] /= params.repetition_penalty
                    else:
                        logits[tid] *= params.repetition_penalty

                    # Presence and frequency penalties
                    logits[tid] -= params.presence_penalty + (params.frequency_penalty * count)

        # 2. Greedy sampling for temperature == 0
        if params.temperature <= 1e-4:
            return int(np.argmax(logits))

        # 3. Apply temperature
        logits = logits / max(1e-4, params.temperature)

        # 4. Top-K filtering
        if params.top_k > 0 and params.top_k < len(logits):
            top_k_indices = np.argpartition(logits, -params.top_k)[-params.top_k:]
            mask = np.ones_like(logits, dtype=bool)
            mask[top_k_indices] = False
            logits[mask] = -np.inf

        # 5. Softmax to probability distribution
        # Stable softmax
        max_logit = np.max(logits)
        if np.isneginf(max_logit):
            # All filtered out; pick random from top_k
            return int(rng.choice(len(logits)))

        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        # 6. Top-P (Nucleus) filtering
        if 0.0 < params.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Mask tokens beyond cumulative threshold
            cutoff_mask = cumulative_probs > params.top_p
            # Always keep at least one token
            cutoff_mask[0] = False

            filtered_indices = sorted_indices[~cutoff_mask]
            filtered_probs = probs[filtered_indices]
            prob_sum = np.sum(filtered_probs)
            if prob_sum > 0:
                filtered_probs = filtered_probs / prob_sum
                return int(rng.choice(filtered_indices, p=filtered_probs))

        # 7. Standard categorical sampling
        return int(rng.choice(len(probs), p=probs))
