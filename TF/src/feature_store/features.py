"""
Feature Store: RFM, Route Affinity, Channel Preferences
Computes customer features for recommendation system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStore:
    """Feature store for computing RFM, route affinity, and channel preferences"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rfm_segments = config.get('feature_store', {}).get('rfm_segments', 
            ['champion', 'loyal', 'at_risk', 'lost'])
        self.channel_preferences = config.get('feature_store', {}).get('channel_preferences',
            ['email', 'sms', 'push', 'in_app'])
        self.route_affinity_window = config.get('feature_store', {}).get('route_affinity_window_days', 90)
    
    def compute_rfm(self, customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RFM (Recency, Frequency, Monetary) segments
        
        Args:
            customers: Customer table with customer_id
            orders: Flight orders with customer_id, order_date, order_value
            
        Returns:
            DataFrame with customer_id, recency, frequency, monetary, rfm_segment
        """
        logger.info("Computing RFM segments...")
        
        # Calculate recency (days since last order)
        last_order = orders.groupby('customer_id')['order_date'].max().reset_index()
        last_order.columns = ['customer_id', 'last_order_date']
        last_order['recency'] = (datetime.now() - pd.to_datetime(last_order['last_order_date'])).dt.days
        
        # Calculate frequency (number of orders in last year)
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        one_year_ago = datetime.now() - timedelta(days=365)
        recent_orders = orders[orders['order_date'] >= one_year_ago]
        frequency = recent_orders.groupby('customer_id').size().reset_index(name='frequency')
        
        # Calculate monetary (total value in last year)
        monetary = recent_orders.groupby('customer_id')['order_value'].sum().reset_index(name='monetary')
        
        # Merge
        rfm = customers[['customer_id']].merge(last_order[['customer_id', 'recency']], on='customer_id', how='left')
        rfm = rfm.merge(frequency, on='customer_id', how='left')
        rfm = rfm.merge(monetary, on='customer_id', how='left')
        
        # Fill NaN with 0
        rfm = rfm.fillna(0)
        
        # Score RFM (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        rfm['r_score'] = pd.to_numeric(rfm['r_score'], errors='coerce').fillna(3)
        rfm['f_score'] = pd.to_numeric(rfm['f_score'], errors='coerce').fillna(3)
        rfm['m_score'] = pd.to_numeric(rfm['m_score'], errors='coerce').fillna(3)
        
        # RFM segment mapping
        def assign_segment(row):
            r, f, m = row['r_score'], row['f_score'], row['m_score']
            if r >= 4 and f >= 4 and m >= 4:
                return 'champion'
            elif r >= 3 and f >= 3:
                return 'loyal'
            elif r <= 2 and f <= 2:
                return 'lost'
            else:
                return 'at_risk'
        
        rfm['rfm_segment'] = rfm.apply(assign_segment, axis=1)
        
        logger.info(f"RFM segments computed: {rfm['rfm_segment'].value_counts().to_dict()}")
        return rfm
    
    def compute_route_affinity(self, orders: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
        """
        Compute route affinity (preferred routes/destinations)
        
        Args:
            orders: Flight orders with customer_id, route, destination
            offers: Offer catalog with route, destination
            
        Returns:
            DataFrame with customer_id, route, destination, affinity_score
        """
        logger.info("Computing route affinity...")
        
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        cutoff_date = datetime.now() - timedelta(days=self.route_affinity_window)
        recent_orders = orders[orders['order_date'] >= cutoff_date]
        
        # Count orders per customer-route
        route_counts = recent_orders.groupby(['customer_id', 'route']).size().reset_index(name='order_count')
        
        # Normalize by total orders per customer
        customer_totals = route_counts.groupby('customer_id')['order_count'].sum().reset_index(name='total_orders')
        route_counts = route_counts.merge(customer_totals, on='customer_id')
        route_counts['affinity_score'] = route_counts['order_count'] / route_counts['total_orders']
        
        # Merge with destination from offers
        if 'destination' in offers.columns and 'route' in offers.columns:
            route_dest = offers[['route', 'destination']].drop_duplicates()
            route_counts = route_counts.merge(route_dest, on='route', how='left')
        
        logger.info(f"Route affinity computed for {len(route_counts['customer_id'].unique())} customers")
        return route_counts[['customer_id', 'route', 'destination', 'affinity_score']]
    
    def compute_channel_preferences(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Compute channel preferences (email, sms, push, in_app)
        
        Args:
            interactions: Interaction data with customer_id, channel, action (click, open, convert)
            
        Returns:
            DataFrame with customer_id, channel, preference_score
        """
        logger.info("Computing channel preferences...")
        
        # Weight actions: convert=3, click=2, open=1
        action_weights = {'convert': 3, 'click': 2, 'open': 1, 'view': 0.5}
        interactions['action_weight'] = interactions['action'].map(action_weights).fillna(0)
        
        # Aggregate by customer and channel
        channel_scores = interactions.groupby(['customer_id', 'channel'])['action_weight'].sum().reset_index(name='score')
        
        # Normalize by total score per customer
        customer_totals = channel_scores.groupby('customer_id')['score'].sum().reset_index(name='total_score')
        channel_scores = channel_scores.merge(customer_totals, on='customer_id')
        channel_scores['preference_score'] = channel_scores['score'] / (channel_scores['total_score'] + 1e-6)
        
        logger.info(f"Channel preferences computed for {len(channel_scores['customer_id'].unique())} customers")
        return channel_scores[['customer_id', 'channel', 'preference_score']]
    
    def get_customer_features(self, customer_id: int, 
                             rfm_df: pd.DataFrame,
                             route_affinity_df: pd.DataFrame,
                             channel_pref_df: pd.DataFrame) -> Dict:
        """
        Get all features for a specific customer
        
        Args:
            customer_id: Customer ID
            rfm_df: RFM DataFrame
            route_affinity_df: Route affinity DataFrame
            channel_pref_df: Channel preferences DataFrame
            
        Returns:
            Dictionary with all customer features
        """
        features = {'customer_id': customer_id}
        
        # RFM features
        rfm_row = rfm_df[rfm_df['customer_id'] == customer_id]
        if not rfm_row.empty:
            features['recency'] = float(rfm_row.iloc[0]['recency'])
            features['frequency'] = float(rfm_row.iloc[0]['frequency'])
            features['monetary'] = float(rfm_row.iloc[0]['monetary'])
            features['rfm_segment'] = rfm_row.iloc[0]['rfm_segment']
        else:
            features['recency'] = 365.0
            features['frequency'] = 0.0
            features['monetary'] = 0.0
            features['rfm_segment'] = 'lost'
        
        # Route affinity (top 3 routes)
        route_rows = route_affinity_df[route_affinity_df['customer_id'] == customer_id].nlargest(3, 'affinity_score')
        features['top_routes'] = route_rows['route'].tolist() if not route_rows.empty else []
        features['top_route_scores'] = route_rows['affinity_score'].tolist() if not route_rows.empty else []
        
        # Channel preferences
        channel_rows = channel_pref_df[channel_pref_df['customer_id'] == customer_id].nlargest(1, 'preference_score')
        features['preferred_channel'] = channel_rows.iloc[0]['channel'] if not channel_rows.empty else 'email'
        features['channel_pref_score'] = float(channel_rows.iloc[0]['preference_score']) if not channel_rows.empty else 0.5
        
        return features

