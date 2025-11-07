"""
Re-ranker: XGBoost-based ranking model
Combines vector scores with business features
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """XGBoost-based re-ranker for combining vector scores with business features"""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize re-ranker
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            feature_names: List of feature names (for interpretability)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_names = feature_names or [
            'vector_score', 'propensity', 'recency', 'budget_fit', 'channel_pref'
        ]
        
        self.model = None
        self.is_trained = False
    
    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              feature_names: Optional[List[str]] = None):
        """
        Train re-ranker
        
        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) binary labels (1 = relevant, 0 = not relevant)
            feature_names: Optional feature names
        """
        logger.info(f"Training re-ranker on {len(X)} samples...")
        
        if feature_names:
            self.feature_names = feature_names
        
        # XGBoost parameters for ranking
        params = {
            'objective': 'binary:logistic',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'eval_metric': 'auc',
            'tree_method': 'hist'
        }
        
        # Train model
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        self.is_trained = True
        logger.info("Re-ranker trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores
        
        Args:
            X: (n_samples, n_features) feature matrix
        
        Returns:
            scores: (n_samples,) relevance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        scores = self.model.predict(dtest)
        return scores
    
    def rerank(self,
              candidates: List[Dict],
              user_features: Optional[Dict] = None) -> List[Dict]:
        """
        Re-rank candidate journeys
        
        Args:
            candidates: List of dicts with 'journey_id', 'vector_score', and other features
            user_features: Optional user features for computing additional features
        
        Returns:
            Re-ranked list of candidates with 'rerank_score'
        """
        if not self.is_trained:
            logger.warning("Model not trained. Returning candidates sorted by vector_score.")
            return sorted(candidates, key=lambda x: x.get('vector_score', 0), reverse=True)
        
        # Extract features
        feature_matrix = []
        for candidate in candidates:
            features = []
            for feat_name in self.feature_names:
                if feat_name in candidate:
                    features.append(candidate[feat_name])
                elif user_features and feat_name in user_features:
                    features.append(user_features[feat_name])
                else:
                    features.append(0.0)  # Default value
            feature_matrix.append(features)
        
        X = np.array(feature_matrix)
        scores = self.predict(X)
        
        # Add rerank scores and sort
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked
    
    def compute_features(self,
                        journey_id: int,
                        vector_score: float,
                        journey_metadata: Dict,
                        user_features: Dict) -> Dict:
        """
        Compute features for a candidate journey
        
        Args:
            journey_id: Journey ID
            vector_score: Vector similarity score
            journey_metadata: Journey metadata (price, region, etc.)
            user_features: User features (RFM, preferences, etc.)
        
        Returns:
            Feature dict
        """
        features = {
            'journey_id': journey_id,
            'vector_score': vector_score,
        }
        
        # Propensity score (simplified: based on RFM segment)
        rfm_segment = user_features.get('rfm_segment', 'lost')
        propensity_map = {'champion': 0.9, 'loyal': 0.7, 'at_risk': 0.5, 'lost': 0.3}
        features['propensity'] = propensity_map.get(rfm_segment, 0.5)
        
        # Recency (inverse of days since last order)
        recency = user_features.get('recency', 365)
        features['recency'] = 1.0 / (1.0 + recency / 30.0)  # Normalize to [0, 1]
        
        # Budget fit (simplified: check if journey price matches user monetary value)
        journey_price = journey_metadata.get('price', 0)
        user_monetary = user_features.get('monetary', 0)
        if user_monetary > 0:
            price_ratio = min(journey_price / user_monetary, 2.0)  # Cap at 2x
            features['budget_fit'] = 1.0 / (1.0 + abs(1.0 - price_ratio))
        else:
            features['budget_fit'] = 0.5
        
        # Channel preference
        preferred_channel = user_features.get('preferred_channel', 'email')
        journey_channel = journey_metadata.get('channel', 'email')
        features['channel_pref'] = 1.0 if preferred_channel == journey_channel else 0.5
        
        return features
    
    def save(self, filepath: str):
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        logger.info(f"Saving re-ranker to {filepath}...")
        self.model.save_model(filepath)
        logger.info("Re-ranker saved successfully")
    
    def load(self, filepath: str):
        """Load model from file"""
        logger.info(f"Loading re-ranker from {filepath}...")
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        self.is_trained = True
        logger.info("Re-ranker loaded successfully")

