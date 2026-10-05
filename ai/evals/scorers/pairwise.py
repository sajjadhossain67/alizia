"""
ALIZIA AI - Elo Rating & Bradley-Terry Pairwise Statistics
Computes Elo ratings, Bradley-Terry likelihood, and non-parametric bootstrap
95% confidence intervals for head-to-head model battles.
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Any, Tuple
from ai.evals.types import PairwiseComparison


class PairwiseRanker:
    """Computes Elo ratings and Bootstrap Confidence Intervals"""

    def __init__(self, initial_elo: float = 1200.0, k_factor: float = 32.0):
        self.initial_elo = initial_elo
        self.k_factor = k_factor

    def compute_elo_ratings(self, comparisons: List[PairwiseComparison]) -> Dict[str, float]:
        """Calculates Elo ratings across all pairwise comparisons"""
        ratings: Dict[str, float] = {}

        for comp in comparisons:
            if comp.model_a not in ratings:
                ratings[comp.model_a] = self.initial_elo
            if comp.model_b not in ratings:
                ratings[comp.model_b] = self.initial_elo

            r_a = ratings[comp.model_a]
            r_b = ratings[comp.model_b]

            # Expected scores
            expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
            expected_b = 1.0 - expected_a

            # Actual score S (1.0 for win, 0.5 for tie, 0.0 for loss)
            if comp.winner == "model_a":
                score_a = 1.0
                score_b = 0.0
            elif comp.winner == "model_b":
                score_a = 0.0
                score_b = 1.0
            else:
                score_a = 0.5
                score_b = 0.5

            # Update ratings
            ratings[comp.model_a] = r_a + self.k_factor * (score_a - expected_a)
            ratings[comp.model_b] = r_b + self.k_factor * (score_b - expected_b)

        return ratings

    @classmethod
    def compute_win_rate_with_bootstrap_ci(
        cls,
        comparisons: List[PairwiseComparison],
        target_model: str,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculates win rate (wins + 0.5*ties / total) and non-parametric
        bootstrap confidence intervals.
        """
        if not comparisons:
            return {"win_rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "total": 0}

        # Filter comparisons involving target_model
        relevant = [
            c for c in comparisons
            if c.model_a == target_model or c.model_b == target_model
        ]
        if not relevant:
            return {"win_rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "total": 0}

        def calculate_score(sublist: List[PairwiseComparison]) -> float:
            points = 0.0
            for c in sublist:
                is_a = (c.model_a == target_model)
                if c.winner == "model_a":
                    points += 1.0 if is_a else 0.0
                elif c.winner == "model_b":
                    points += 0.0 if is_a else 1.0
                else:
                    points += 0.5
            return (points / len(sublist)) * 100.0

        sample_win_rate = calculate_score(relevant)

        # Bootstrap resampling
        bootstrap_scores: List[float] = []
        n_samples = len(relevant)
        random.seed(42)  # Pinned seed for reproducibility

        for _ in range(n_bootstraps):
            resampled = [random.choice(relevant) for _ in range(n_samples)]
            bootstrap_scores.append(calculate_score(resampled))

        bootstrap_scores.sort()
        alpha = (1.0 - confidence_level) / 2.0
        lower_idx = int(alpha * n_bootstraps)
        upper_idx = int((1.0 - alpha) * n_bootstraps)

        ci_lower = bootstrap_scores[min(lower_idx, n_bootstraps - 1)]
        ci_upper = bootstrap_scores[min(upper_idx, n_bootstraps - 1)]

        return {
            "win_rate": round(sample_win_rate, 2),
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            "total_battles": n_samples
        }
