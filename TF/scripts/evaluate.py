"""
Evaluate Recommendation System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_retriever import FAISSRetriever
from src.reranker.reranker import Reranker
from src.evaluation.metrics import evaluate_recommendations, compute_ann_recall
from src.serving.pipeline import RecommendationPipeline


def generate_ground_truth(interactions_df: pd.DataFrame, k: int = 10):
    """Generate ground truth from interactions"""
    # For each user, get top-k most interacted journeys
    user_journeys = interactions_df.groupby('user_id').apply(
        lambda x: x.nlargest(k, 'label')['journey_id'].tolist()
    ).to_dict()
    
    return user_journeys


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("Loading models...")
    checkpoint = torch.load('models/two_tower_model.pt', map_location='cpu')
    model = TwoTowerModel(
        num_users=checkpoint['num_users'],
        num_journeys=checkpoint['num_journeys'],
        **checkpoint['config']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load FAISS index
    retriever = FAISSRetriever(embedding_dim=config['model']['embedding_dim'])
    retriever.load('models/faiss_index.index')
    
    # Load re-ranker
    reranker = Reranker()
    reranker.load('models/reranker.model')
    
    # Load data
    print("Loading data...")
    interactions = pd.read_csv('data/interactions.csv')
    
    # Generate ground truth
    print("Generating ground truth...")
    ground_truth = generate_ground_truth(interactions, k=10)
    
    # Create pipeline
    pipeline = RecommendationPipeline(
        embedding_model=model,
        faiss_retriever=retriever,
        reranker=reranker,
        feature_store=None  # Simplified for evaluation
    )
    
    # Evaluate on sample users
    print("Evaluating recommendations...")
    test_users = list(ground_truth.keys())[:100]  # Sample 100 users
    
    recommendations = {}
    for user_id in test_users:
        try:
            recs = pipeline.recommend(user_id, top_k=10)
            recommendations[user_id] = [r['journey_id'] for r in recs]
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            recommendations[user_id] = []
    
    # Compute metrics
    metrics = evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=ground_truth,
        k=10
    )
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Precision@10: {metrics['precision@k']:.4f}")
    print(f"Recall@10: {metrics['recall@k']:.4f}")
    print(f"NDCG@10: {metrics['ndcg@k']:.4f}")
    print(f"MAP@10: {metrics['map@k']:.4f}")
    print(f"Number of users: {metrics['num_users']}")
    print("="*50)


if __name__ == '__main__':
    main()

