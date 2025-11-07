"""
Serve Recommendations (CLI)
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_retriever import FAISSRetriever
from src.reranker.reranker import Reranker
from src.serving.pipeline import RecommendationPipeline


def main():
    parser = argparse.ArgumentParser(description='Serve recommendations')
    parser.add_argument('--user_id', type=int, required=True, help='User ID')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--filter_region', type=str, default=None, help='Filter by region')
    parser.add_argument('--filter_channel', type=str, default=None, help='Filter by channel')
    
    args = parser.parse_args()
    
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
    
    # Create pipeline
    pipeline = RecommendationPipeline(
        embedding_model=model,
        faiss_retriever=retriever,
        reranker=reranker,
        feature_store=None
    )
    
    # Build filter dict
    filter_dict = {}
    if args.filter_region:
        filter_dict['region'] = args.filter_region
    if args.filter_channel:
        filter_dict['channel'] = args.filter_channel
    
    # Get recommendations
    print(f"\nGenerating recommendations for user {args.user_id}...")
    recommendations = pipeline.recommend(
        user_id=args.user_id,
        top_k=args.top_k,
        filter_dict=filter_dict if filter_dict else None
    )
    
    # Display results
    print(f"\nTop {len(recommendations)} Recommendations:")
    print("-" * 80)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. Journey ID: {rec['journey_id']}")
        print(f"   Vector Score: {rec['vector_score']:.4f}")
        print(f"   Re-rank Score: {rec.get('rerank_score', 0):.4f}")
        print(f"   Propensity: {rec.get('propensity', 0):.4f}")
        print(f"   Budget Fit: {rec.get('budget_fit', 0):.4f}")
        print()


if __name__ == '__main__':
    main()

