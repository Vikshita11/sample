"""
Two-Tower Embedding Model
User tower and Journey tower for collaborative filtering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserTower(nn.Module):
    """User embedding tower"""
    
    def __init__(self, 
                 num_users: int,
                 embedding_dim: int = 128,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.2):
        super(UserTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        
        # User ID embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Feature layers
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final projection to embedding_dim
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,)
        Returns:
            user_embeddings: (batch_size, embedding_dim)
        """
        user_emb = self.user_embedding(user_ids)
        return self.layers(user_emb)


class JourneyTower(nn.Module):
    """Journey/Offer embedding tower"""
    
    def __init__(self,
                 num_journeys: int,
                 embedding_dim: int = 128,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.2,
                 num_features: int = 0):
        super(JourneyTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_journeys = num_journeys
        
        # Journey ID embedding
        self.journey_embedding = nn.Embedding(num_journeys, embedding_dim)
        
        # Feature layers
        layers = []
        input_dim = embedding_dim + num_features  # ID embedding + features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final projection to embedding_dim
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, journey_ids: torch.Tensor, journey_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            journey_ids: (batch_size,)
            journey_features: (batch_size, num_features) optional
        Returns:
            journey_embeddings: (batch_size, embedding_dim)
        """
        journey_emb = self.journey_embedding(journey_ids)
        
        if journey_features is not None:
            x = torch.cat([journey_emb, journey_features], dim=1)
        else:
            x = journey_emb
        
        return self.layers(x)


class TwoTowerModel(nn.Module):
    """Two-tower model for user-journey matching"""
    
    def __init__(self,
                 num_users: int,
                 num_journeys: int,
                 embedding_dim: int = 128,
                 user_tower_hidden_dims: list = [256, 128],
                 journey_tower_hidden_dims: list = [256, 128],
                 dropout: float = 0.2,
                 journey_num_features: int = 0):
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=embedding_dim,
            hidden_dims=user_tower_hidden_dims,
            dropout=dropout
        )
        
        self.journey_tower = JourneyTower(
            num_journeys=num_journeys,
            embedding_dim=embedding_dim,
            hidden_dims=journey_tower_hidden_dims,
            dropout=dropout,
            num_features=journey_num_features
        )
    
    def forward(self, 
                user_ids: torch.Tensor,
                journey_ids: torch.Tensor,
                journey_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            user_ids: (batch_size,)
            journey_ids: (batch_size,)
            journey_features: (batch_size, num_features) optional
        Returns:
            user_embeddings: (batch_size, embedding_dim)
            journey_embeddings: (batch_size, embedding_dim)
        """
        user_emb = self.user_tower(user_ids)
        journey_emb = self.journey_tower(journey_ids, journey_features)
        
        return user_emb, journey_emb
    
    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings"""
        return self.user_tower(user_ids)
    
    def get_journey_embedding(self, 
                             journey_ids: torch.Tensor,
                             journey_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get journey embeddings"""
        return self.journey_tower(journey_ids, journey_features)
    
    def compute_similarity(self, 
                          user_emb: torch.Tensor,
                          journey_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between user and journey embeddings
        
        Args:
            user_emb: (batch_size, embedding_dim) or (embedding_dim,)
            journey_emb: (batch_size, embedding_dim) or (num_journeys, embedding_dim)
        
        Returns:
            similarities: (batch_size, num_journeys) or scalar
        """
        # Normalize embeddings
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        journey_emb = F.normalize(journey_emb, p=2, dim=-1)
        
        # Compute dot product (cosine similarity)
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0)
        if journey_emb.dim() == 1:
            journey_emb = journey_emb.unsqueeze(0)
        
        return torch.matmul(user_emb, journey_emb.t())


def train_two_tower(model: TwoTowerModel,
                   train_loader,
                   num_epochs: int = 10,
                   learning_rate: float = 0.001,
                   device: str = 'cpu'):
    """
    Train two-tower model with contrastive loss
    
    Args:
        model: TwoTowerModel instance
        train_loader: DataLoader with (user_id, journey_id, label)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            journey_ids = batch['journey_id'].to(device)
            labels = batch['label'].float().to(device)
            
            # Get embeddings
            user_emb, journey_emb = model(user_ids, journey_ids)
            
            # Compute similarity (dot product)
            similarity = (user_emb * journey_emb).sum(dim=1)
            
            # Loss
            loss = criterion(similarity, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

