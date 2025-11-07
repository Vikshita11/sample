"""
Build FAISS Index for Candidate Generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from src.retrieval.faiss_retriever import FAISSRetriever


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    faiss_config = config['faiss']
    model_config = config['model']
    
    # Load journey embeddings
    print("Loading journey embeddings...")
    journey_embeddings = np.load('models/journey_embeddings.npy')
    
    # Load journey metadata
    try:
        offers = pd.read_csv('data/offer_catalog.csv')
        journey_ids = offers['journey_id'].tolist()
        
        # Create metadata dict
        metadata = {}
        for _, row in offers.iterrows():
            metadata[row['journey_id']] = {
                'region': row.get('region', 'unknown'),
                'price': row.get('price', 0),
                'channel': row.get('channel', 'email')
            }
    except FileNotFoundError:
        print("Warning: offer_catalog.csv not found. Using sequential journey IDs.")
        journey_ids = list(range(len(journey_embeddings)))
        metadata = {}
    
    print(f"Building index for {len(journey_embeddings)} journeys...")
    
    # Create retriever
    retriever = FAISSRetriever(
        embedding_dim=model_config['embedding_dim'],
        index_type=faiss_config['index_type'],
        hnsw_m=faiss_config['hnsw_m'],
        hnsw_ef_construction=faiss_config['hnsw_ef_construction'],
        hnsw_ef_search=faiss_config['hnsw_ef_search'],
        nprobe=faiss_config['nprobe']
    )
    
    # Build index
    retriever.build_index(
        journey_embeddings=journey_embeddings,
        journey_ids=journey_ids,
        metadata=metadata
    )
    
    # Save index
    os.makedirs('models', exist_ok=True)
    retriever.save('models/faiss_index.index')
    
    print("Index saved to models/faiss_index.index")
    
    # Test search
    print("\nTesting search...")
    test_user_emb = journey_embeddings[0]  # Use first journey as test
    results, scores = retriever.search(test_user_emb, top_k=5)
    print(f"Top 5 results: {results}")
    print(f"Scores: {[f'{s:.4f}' for s in scores]}")


if __name__ == '__main__':
    main()

