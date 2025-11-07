"""
Train Two-Tower Embedding Model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

from src.models.two_tower import TwoTowerModel, train_two_tower
from src.data_generator import generate_sample_data


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
            'label': row['label']  # 1 for positive, 0 for negative
        }


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Load or generate data
    print("Loading interaction data...")
    try:
        interactions = pd.read_csv('data/interactions.csv')
    except FileNotFoundError:
        print("Generating sample data...")
        generate_sample_data()
        interactions = pd.read_csv('data/interactions.csv')
    
    # Get unique IDs
    num_users = interactions['user_id'].nunique()
    num_journeys = interactions['journey_id'].nunique()
    
    print(f"Training on {len(interactions)} interactions")
    print(f"Users: {num_users}, Journeys: {num_journeys}")
    
    # Create model
    model = TwoTowerModel(
        num_users=num_users,
        num_journeys=num_journeys,
        embedding_dim=model_config['embedding_dim'],
        user_tower_hidden_dims=model_config['user_tower_hidden_dims'],
        journey_tower_hidden_dims=model_config['journey_tower_hidden_dims'],
        dropout=model_config['dropout']
    )
    
    # Create dataset and dataloader
    dataset = InteractionDataset(interactions)
    train_loader = DataLoader(
        dataset,
        batch_size=model_config['batch_size'],
        shuffle=True
    )
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    model = train_two_tower(
        model=model,
        train_loader=train_loader,
        num_epochs=model_config['num_epochs'],
        learning_rate=model_config['learning_rate'],
        device=device
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_journeys': num_journeys,
        'config': model_config
    }, 'models/two_tower_model.pt')
    
    print("Model saved to models/two_tower_model.pt")
    
    # Generate and save embeddings
    print("Generating embeddings...")
    model.eval()
    
    # User embeddings
    user_ids = torch.arange(num_users, dtype=torch.long)
    with torch.no_grad():
        user_embeddings = model.get_user_embedding(user_ids)
    
    np.save('models/user_embeddings.npy', user_embeddings.numpy())
    
    # Journey embeddings
    journey_ids = torch.arange(num_journeys, dtype=torch.long)
    with torch.no_grad():
        journey_embeddings = model.get_journey_embedding(journey_ids)
    
    np.save('models/journey_embeddings.npy', journey_embeddings.numpy())
    
    print("Embeddings saved to models/")


if __name__ == '__main__':
    main()

