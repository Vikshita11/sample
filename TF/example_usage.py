"""
Example Usage: Complete Recommendation Pipeline
Demonstrates the full workflow from data generation to serving recommendations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import numpy as np
import pandas as pd

from src.data_generator import generate_sample_data
from src.feature_store.features import FeatureStore
from src.models.two_tower import TwoTowerModel, train_two_tower
from src.retrieval.faiss_retriever import FAISSRetriever
from src.reranker.reranker import Reranker
from src.serving.pipeline import RecommendationPipeline
from src.evaluation.metrics import evaluate_recommendations
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """Dataset for user-journey interactions"""
    
    def __init__(self, interactions_df: pd.DataFrame):
        self.interactions = interactions_df
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        return {
            'user_id': row['user_id'],
            'journey_id': row['journey_id'],
            'label': row['label']
        }


def main():
    print("="*80)
    print("Hybrid Two-Stage Recommendation System - Example Usage")
    print("="*80)
    
    # Load config
    print("\n1. Loading configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Generate sample data
    print("\n2. Generating sample data...")
    if not os.path.exists('data/interactions.csv'):
        generate_sample_data(num_users=1000, num_journeys=5000, num_interactions=10000)
    else:
        print("   Data already exists, skipping generation...")
    
    # Step 2: Compute features
    print("\n3. Computing features (RFM, route affinity, channel preferences)...")
    customers = pd.read_csv('data/customers.csv')
    orders = pd.read_csv('data/flight_orders.csv')
    offers = pd.read_csv('data/offer_catalog.csv')
    interactions = pd.read_csv('data/interactions.csv')
    
    feature_store = FeatureStore(config)
    rfm_df = feature_store.compute_rfm(customers, orders)
    route_affinity_df = feature_store.compute_route_affinity(orders, offers)
    
    # Step 3: Train two-tower model
    print("\n4. Training two-tower embedding model...")
    num_users = interactions['user_id'].nunique()
    num_journeys = interactions['journey_id'].nunique()
    
    model = TwoTowerModel(
        num_users=num_users,
        num_journeys=num_journeys,
        embedding_dim=config['model']['embedding_dim'],
        user_tower_hidden_dims=config['model']['user_tower_hidden_dims'],
        journey_tower_hidden_dims=config['model']['journey_tower_hidden_dims'],
        dropout=config['model']['dropout']
    )
    
    # Create dataloader
    dataset = InteractionDataset(interactions)
    train_loader = DataLoader(
        dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True
    )
    
    # Train (quick training for demo)
    print("   Training model (this may take a few minutes)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_two_tower(
        model=model,
        train_loader=train_loader,
        num_epochs=config['model']['num_epochs'],
        learning_rate=config['model']['learning_rate'],
        device=device
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_journeys': num_journeys,
        'config': config['model']
    }, 'models/two_tower_model.pt')
    print("   Model saved to models/two_tower_model.pt")
    
    # Generate embeddings
    print("\n5. Generating embeddings...")
    model.eval()
    with torch.no_grad():
        user_ids = torch.arange(num_users, dtype=torch.long)
        journey_ids = torch.arange(num_journeys, dtype=torch.long)
        
        user_embeddings = model.get_user_embedding(user_ids).numpy()
        journey_embeddings = model.get_journey_embedding(journey_ids).numpy()
    
    np.save('models/user_embeddings.npy', user_embeddings)
    np.save('models/journey_embeddings.npy', journey_embeddings)
    print("   Embeddings saved to models/")
    
    # Step 4: Build FAISS index
    print("\n6. Building FAISS index...")
    retriever = FAISSRetriever(
        embedding_dim=config['model']['embedding_dim'],
        index_type=config['faiss']['index_type'],
        hnsw_m=config['faiss']['hnsw_m'],
        hnsw_ef_construction=config['faiss']['hnsw_ef_construction'],
        hnsw_ef_search=config['faiss']['hnsw_ef_search'],
        nprobe=config['faiss']['nprobe']
    )
    
    journey_ids_list = offers['journey_id'].tolist() if 'journey_id' in offers.columns else list(range(num_journeys))
    metadata = {}
    for _, row in offers.iterrows():
        metadata[row['journey_id']] = {
            'region': row.get('region', 'unknown'),
            'price': row.get('price', 0),
            'channel': row.get('channel', 'email')
        }
    
    retriever.build_index(
        journey_embeddings=journey_embeddings,
        journey_ids=journey_ids_list,
        metadata=metadata
    )
    
    retriever.save('models/faiss_index.index')
    print("   Index saved to models/faiss_index.index")
    
    # Step 5: Train re-ranker
    print("\n7. Training re-ranker...")
    # Generate synthetic training data
    num_samples = 10000
    train_data = pd.DataFrame({
        'vector_score': np.random.random(num_samples),
        'propensity': np.random.random(num_samples),
        'recency': np.random.random(num_samples),
        'budget_fit': np.random.random(num_samples),
        'channel_pref': np.random.random(num_samples),
        'label': (np.random.random(num_samples) > 0.5).astype(int)
    })
    
    feature_cols = config['reranker']['features']
    X = train_data[feature_cols].values
    y = train_data['label'].values
    
    reranker = Reranker(
        n_estimators=config['reranker']['n_estimators'],
        max_depth=config['reranker']['max_depth'],
        learning_rate=config['reranker']['learning_rate'],
        feature_names=feature_cols
    )
    
    reranker.train(X, y, feature_names=feature_cols)
    reranker.save('models/reranker.model')
    print("   Re-ranker saved to models/reranker.model")
    
    # Step 6: Serve recommendations
    print("\n8. Serving recommendations...")
    pipeline = RecommendationPipeline(
        embedding_model=model,
        faiss_retriever=retriever,
        reranker=reranker,
        feature_store=feature_store
    )
    
    # Test with a sample user
    test_user_id = 1
    recommendations = pipeline.recommend(
        user_id=test_user_id,
        top_k=10
    )
    
    print(f"\n   Recommendations for user {test_user_id}:")
    print("-" * 80)
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. Journey ID: {rec['journey_id']}")
        print(f"      Vector Score: {rec['vector_score']:.4f}")
        print(f"      Re-rank Score: {rec.get('rerank_score', 0):.4f}")
        print(f"      Propensity: {rec.get('propensity', 0):.4f}")
        print()
    
    print("\n" + "="*80)
    print("Pipeline setup complete!")
    print("="*80)
    print("\nNext steps:")
    print("  - Run 'python scripts/evaluate.py' for full evaluation")
    print("  - Run 'python scripts/serve.py --user_id 1 --top_k 10' for serving")
    print("  - Check models/ directory for saved models and indexes")


if __name__ == '__main__':
    main()

