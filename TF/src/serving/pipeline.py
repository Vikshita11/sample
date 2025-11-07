"""
Serving Pipeline: Online recommendation serving with suppression/cadence rules
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """End-to-end recommendation pipeline"""
    
    def __init__(self,
                 embedding_model,
                 faiss_retriever,
                 reranker,
                 feature_store,
                 suppression_rules: Optional[Dict] = None,
                 cadence_rules: Optional[Dict] = None):
        """
        Initialize recommendation pipeline
        
        Args:
            embedding_model: Trained TwoTowerModel
            faiss_retriever: FAISSRetriever instance
            reranker: Reranker instance
            feature_store: FeatureStore instance
            suppression_rules: Dict with suppression logic
            cadence_rules: Dict with cadence/contact frequency rules
        """
        self.embedding_model = embedding_model
        self.faiss_retriever = faiss_retriever
        self.reranker = reranker
        self.feature_store = feature_store
        self.suppression_rules = suppression_rules or {}
        self.cadence_rules = cadence_rules or {}
        
        # Track user contact history (in production, use Redis/DB)
        self.user_contact_history = {}
    
    def recommend(self,
                 user_id: int,
                 top_k: int = 10,
                 filter_dict: Optional[Dict] = None,
                 user_features: Optional[Dict] = None,
                 journey_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations to return
            filter_dict: Optional metadata filters for pre-filtering
            user_features: Optional pre-computed user features
            journey_metadata: Optional dict mapping journey_id to metadata
        
        Returns:
            List of recommended journeys with scores
        """
        # Check suppression rules
        if self._is_suppressed(user_id, user_features):
            logger.info(f"User {user_id} is suppressed. Returning empty recommendations.")
            return []
        
        # Check cadence rules
        if not self._check_cadence(user_id):
            logger.info(f"User {user_id} contacted too recently. Skipping.")
            return []
        
        # Get user embedding
        user_emb = self._get_user_embedding(user_id)
        if user_emb is None:
            logger.warning(f"Could not get embedding for user {user_id}")
            return []
        
        # Candidate generation (ANN search)
        candidate_ids, vector_scores = self.faiss_retriever.search(
            user_emb,
            top_k=1000,  # Retrieve more for re-ranking
            filter_dict=filter_dict
        )
        
        if len(candidate_ids) == 0:
            return []
        
        # Get user features if not provided
        if user_features is None:
            user_features = self._get_user_features(user_id)
        
        # Prepare candidates for re-ranking
        candidates = []
        for journey_id, vector_score in zip(candidate_ids, vector_scores):
            # Get journey metadata
            journey_meta = journey_metadata.get(journey_id, {}) if journey_metadata else {}
            
            # Compute features
            features = self.reranker.compute_features(
                journey_id=journey_id,
                vector_score=vector_score,
                journey_metadata=journey_meta,
                user_features=user_features
            )
            
            # Apply suppression rules per journey
            if not self._is_journey_suppressed(journey_id, user_features, journey_meta):
                candidates.append(features)
        
        # Re-rank
        reranked = self.reranker.rerank(candidates, user_features)
        
        # Return top-k
        recommendations = reranked[:top_k]
        
        # Update contact history
        self._update_contact_history(user_id)
        
        return recommendations
    
    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user embedding from model"""
        try:
            self.embedding_model.eval()
            with torch.no_grad():
                user_tensor = torch.tensor([user_id], dtype=torch.long)
                user_emb = self.embedding_model.get_user_embedding(user_tensor)
                return user_emb.numpy()[0]
        except Exception as e:
            logger.error(f"Error getting user embedding: {e}")
            return None
    
    def _get_user_features(self, user_id: int) -> Dict:
        """Get user features from feature store"""
        # This would typically query the feature store
        # For now, return default features
        return {
            'customer_id': user_id,
            'recency': 30.0,
            'frequency': 2.0,
            'monetary': 1000.0,
            'rfm_segment': 'loyal',
            'preferred_channel': 'email',
            'channel_pref_score': 0.7
        }
    
    def _is_suppressed(self, user_id: int, user_features: Optional[Dict]) -> bool:
        """Check if user should be suppressed"""
        if not self.suppression_rules:
            return False
        
        # Check consent
        if self.suppression_rules.get('require_consent', False):
            if user_features and not user_features.get('consent', True):
                return True
        
        # Check opt-out
        if user_features and user_features.get('opt_out', False):
            return True
        
        return False
    
    def _is_journey_suppressed(self, 
                               journey_id: int,
                               user_features: Dict,
                               journey_metadata: Dict) -> bool:
        """Check if specific journey should be suppressed for user"""
        # Example: Suppress if journey region doesn't match user preference
        if 'region' in journey_metadata and 'preferred_region' in user_features:
            if journey_metadata['region'] != user_features['preferred_region']:
                return False  # Not suppressed, just filtered
        
        return False
    
    def _check_cadence(self, user_id: int) -> bool:
        """Check if user can be contacted based on cadence rules"""
        if not self.cadence_rules:
            return True
        
        if user_id not in self.user_contact_history:
            return True
        
        last_contact = self.user_contact_history[user_id]
        min_days = self.cadence_rules.get('min_days_between_contacts', 7)
        
        days_since = (datetime.now() - last_contact).days
        return days_since >= min_days
    
    def _update_contact_history(self, user_id: int):
        """Update contact history for cadence tracking"""
        self.user_contact_history[user_id] = datetime.now()
    
    def batch_recommend(self,
                       user_ids: List[int],
                       top_k: int = 10,
                       filter_dict: Optional[Dict] = None) -> Dict[int, List[Dict]]:
        """
        Generate recommendations for multiple users (batch mode)
        
        Args:
            user_ids: List of user IDs
            top_k: Number of recommendations per user
            filter_dict: Optional metadata filters
        
        Returns:
            Dict mapping user_id to list of recommendations
        """
        results = {}
        
        for user_id in user_ids:
            try:
                recommendations = self.recommend(
                    user_id=user_id,
                    top_k=top_k,
                    filter_dict=filter_dict
                )
                results[user_id] = recommendations
            except Exception as e:
                logger.error(f"Error recommending for user {user_id}: {e}")
                results[user_id] = []
        
        return results

