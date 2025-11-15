"""
metrics_eval.py

Module for computing evaluation metrics (Precision@K, Recall@K, MAP@K)
for recommender systems, especially those using implicit feedback data.

Usage Example:
--------------
from metrics_eval import precision_at_k, recall_at_k, map_at_k

metrics = {
    "Precision@10": precision_at_k(predicted, ground_truth, k=10),
    "Recall@10": recall_at_k(predicted, ground_truth, k=10),
    "MAP@10": map_at_k(predicted, ground_truth, k=10)
}
"""

import numpy as np
from typing import Dict, List, Set


def precision_at_k(predicted: Dict[int, List[int]], ground_truth: Dict[int, Set[int]], k: int = 10) -> float:
    """Compute average Precision@K across all users."""
    precisions = []
    for user, preds in predicted.items():
        if user not in ground_truth or not ground_truth[user]:
            continue
        preds_at_k = preds[:k]
        hits = len(set(preds_at_k) & ground_truth[user])
        precisions.append(hits / k)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(predicted: Dict[int, List[int]], ground_truth: Dict[int, Set[int]], k: int = 10) -> float:
    """Compute average Recall@K across all users."""
    recalls = []
    for user, preds in predicted.items():
        if user not in ground_truth or not ground_truth[user]:
            continue
        preds_at_k = preds[:k]
        hits = len(set(preds_at_k) & ground_truth[user])
        recalls.append(hits / len(ground_truth[user]))
    return float(np.mean(recalls)) if recalls else 0.0


def map_at_k(predicted: Dict[int, List[int]], ground_truth: Dict[int, Set[int]], k: int = 10) -> float:
    """Compute Mean Average Precision@K (MAP@K)."""
    avg_precisions = []
    for user, preds in predicted.items():
        if user not in ground_truth or not ground_truth[user]:
            continue
        hits = 0
        sum_precisions = 0
        for i, p in enumerate(preds[:k], start=1):
            if p in ground_truth[user]:
                hits += 1
                sum_precisions += hits / i
        avg_precisions.append(sum_precisions / min(len(ground_truth[user]), k))
    return float(np.mean(avg_precisions)) if avg_precisions else 0.0


def evaluate_all(predicted: Dict[int, List[int]], ground_truth: Dict[int, Set[int]], k: int = 10) -> Dict[str, float]:
    """Return a dictionary of all 3 metrics at K."""
    return {
        f"Precision@{k}": precision_at_k(predicted, ground_truth, k),
        f"Recall@{k}": recall_at_k(predicted, ground_truth, k),
        f"MAP@{k}": map_at_k(predicted, ground_truth, k)
    }
