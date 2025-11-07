"""
Evaluation Metrics: Precision@K, Recall@K, NDCG@K, MAP
"""

import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Precision@K
    
    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant (ground truth) item IDs
        k: Top-k to consider
    
    Returns:
        Precision@K score
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    if len(recommended_k) == 0:
        return 0.0
    
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(recommended_k)


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Recall@K
    
    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant (ground truth) item IDs
        k: Top-k to consider
    
    Returns:
        Recall@K score
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    if len(relevant_set) == 0:
        return 0.0
    
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set)


def dcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Compute DCG@K (Discounted Cumulative Gain)
    
    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant (ground truth) item IDs
        k: Top-k to consider
    
    Returns:
        DCG@K score
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            # Relevance = 1 if relevant, 0 otherwise
            rel = 1.0
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    return dcg


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant (ground truth) item IDs
        k: Top-k to consider
    
    Returns:
        NDCG@K score (0-1)
    """
    dcg = dcg_at_k(recommended, relevant, k)
    
    # Ideal DCG (all relevant items at top)
    ideal_recommended = relevant[:k] + [item for item in recommended if item not in relevant]
    idcg = dcg_at_k(ideal_recommended, relevant, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def map_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Compute MAP@K (Mean Average Precision)
    
    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant (ground truth) item IDs
        k: Top-k to consider
    
    Returns:
        MAP@K score
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    if len(relevant_set) == 0:
        return 0.0
    
    precisions = []
    hits = 0
    
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            hits += 1
            precisions.append(hits / (i + 1))
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(relevant_set)


def evaluate_recommendations(recommendations: Dict[int, List[int]],
                           ground_truth: Dict[int, List[int]],
                           k: int = 10) -> Dict[str, float]:
    """
    Evaluate recommendations for multiple users
    
    Args:
        recommendations: Dict mapping user_id to list of recommended journey_ids
        ground_truth: Dict mapping user_id to list of relevant journey_ids
        k: Top-k to evaluate
    
    Returns:
        Dictionary of metrics
    """
    precisions = []
    recalls = []
    ndcgs = []
    maps = []
    
    for user_id in recommendations:
        if user_id not in ground_truth:
            continue
        
        recommended = recommendations[user_id]
        relevant = ground_truth[user_id]
        
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        maps.append(map_at_k(recommended, relevant, k))
    
    metrics = {
        'precision@k': np.mean(precisions) if precisions else 0.0,
        'recall@k': np.mean(recalls) if recalls else 0.0,
        'ndcg@k': np.mean(ndcgs) if ndcgs else 0.0,
        'map@k': np.mean(maps) if maps else 0.0,
        'num_users': len(precisions)
    }
    
    return metrics


def compute_ann_recall(ann_results: List[int],
                      exact_results: List[int],
                      k: int) -> float:
    """
    Compute ANN recall (how many exact top-K appear in ANN results)
    
    Args:
        ann_results: Results from ANN search
        exact_results: Exact (ground truth) top-K results
        k: Top-k to consider
    
    Returns:
        Recall score
    """
    ann_set = set(ann_results[:k])
    exact_set = set(exact_results[:k])
    
    if len(exact_set) == 0:
        return 0.0
    
    intersection = ann_set & exact_set
    return len(intersection) / len(exact_set)

