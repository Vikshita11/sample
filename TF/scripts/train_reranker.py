"""
Train Re-ranker Model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from src.reranker.reranker import Reranker
from src.feature_store.features import FeatureStore


def generate_training_data(num_samples: int = 10000):
    """Generate synthetic training data for re-ranker"""
    np.random.seed(42)
    
    data = []
    for _ in range(num_samples):
        # Features
        vector_score = np.random.random()
        propensity = np.random.random()
        recency = np.random.random()
        budget_fit = np.random.random()
        channel_pref = np.random.random()
        
        # Label (1 if relevant, 0 otherwise)
        # Higher scores = more likely to be relevant
        relevance_prob = (vector_score * 0.3 + propensity * 0.3 + 
                         recency * 0.2 + budget_fit * 0.1 + channel_pref * 0.1)
        label = 1 if relevance_prob > 0.5 else 0
        
        data.append({
            'vector_score': vector_score,
            'propensity': propensity,
            'recency': recency,
            'budget_fit': budget_fit,
            'channel_pref': channel_pref,
            'label': label
        })
    
    return pd.DataFrame(data)


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    reranker_config = config['reranker']
    
    # Generate or load training data
    print("Preparing training data...")
    try:
        train_data = pd.read_csv('data/reranker_training.csv')
    except FileNotFoundError:
        print("Generating synthetic training data...")
        train_data = generate_training_data(num_samples=10000)
        os.makedirs('data', exist_ok=True)
        train_data.to_csv('data/reranker_training.csv', index=False)
    
    # Prepare features and labels
    feature_cols = reranker_config['features']
    X = train_data[feature_cols].values
    y = train_data['label'].values
    
    print(f"Training on {len(X)} samples with {len(feature_cols)} features")
    
    # Create and train re-ranker
    reranker = Reranker(
        n_estimators=reranker_config['n_estimators'],
        max_depth=reranker_config['max_depth'],
        learning_rate=reranker_config['learning_rate'],
        feature_names=feature_cols
    )
    
    reranker.train(X, y, feature_names=feature_cols)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    reranker.save('models/reranker.model')
    
    print("Re-ranker saved to models/reranker.model")
    
    # Test prediction
    print("\nTesting re-ranker...")
    test_features = np.array([[0.8, 0.7, 0.6, 0.9, 0.8]])
    score = reranker.predict(test_features)[0]
    print(f"Test prediction score: {score:.4f}")


if __name__ == '__main__':
    main()

