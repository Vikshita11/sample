"""
Generate Sample Data for Testing
"""

import pandas as pd
import numpy as np
import os


def generate_sample_data(num_users: int = 1000,
                         num_journeys: int = 5000,
                         num_interactions: int = 10000):
    """Generate sample data for training and testing"""
    
    np.random.seed(42)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate customers
    print("Generating customers...")
    customers = pd.DataFrame({
        'customer_id': range(1, num_users + 1),
        'age': np.random.randint(18, 80, num_users),
        'region': np.random.choice(['EU', 'US', 'APAC', 'LATAM'], num_users),
        'consent': np.random.choice([True, False], num_users, p=[0.9, 0.1])
    })
    customers.to_csv('data/customers.csv', index=False)
    
    # Generate flight orders
    print("Generating flight orders...")
    num_orders = num_interactions // 2
    flight_orders = pd.DataFrame({
        'order_id': range(1, num_orders + 1),
        'customer_id': np.random.randint(1, num_users + 1, num_orders),
        'journey_id': np.random.randint(1, num_journeys + 1, num_orders),
        'order_date': pd.date_range('2023-01-01', periods=num_orders, freq='D'),
        'order_value': np.random.uniform(500, 5000, num_orders)
    })
    flight_orders.to_csv('data/flight_orders.csv', index=False)
    
    # Generate offer catalog
    print("Generating offer catalog...")
    regions = ['EU', 'US', 'APAC', 'LATAM']
    channels = ['email', 'sms', 'push', 'in_app']
    
    offer_catalog = pd.DataFrame({
        'journey_id': range(1, num_journeys + 1),
        'route': [f"Route_{i}" for i in range(1, num_journeys + 1)],
        'destination': [f"Dest_{i}" for i in range(1, num_journeys + 1)],
        'region': np.random.choice(regions, num_journeys),
        'price': np.random.uniform(300, 3000, num_journeys),
        'channel': np.random.choice(channels, num_journeys)
    })
    offer_catalog.to_csv('data/offer_catalog.csv', index=False)
    
    # Generate interactions (positive and negative)
    print("Generating interactions...")
    # Positive interactions (bookings, clicks)
    positive_interactions = pd.DataFrame({
        'user_id': np.random.randint(1, num_users + 1, num_interactions // 2),
        'journey_id': np.random.randint(1, num_journeys + 1, num_interactions // 2),
        'action': np.random.choice(['book', 'click'], num_interactions // 2),
        'label': 1
    })
    
    # Negative interactions (random pairs)
    negative_interactions = pd.DataFrame({
        'user_id': np.random.randint(1, num_users + 1, num_interactions // 2),
        'journey_id': np.random.randint(1, num_journeys + 1, num_interactions // 2),
        'action': 'view',
        'label': 0
    })
    
    # Ensure no duplicate positive pairs
    positive_interactions = positive_interactions.drop_duplicates(['user_id', 'journey_id'])
    
    interactions = pd.concat([positive_interactions, negative_interactions], ignore_index=True)
    interactions = interactions.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    interactions.to_csv('data/interactions.csv', index=False)
    
    print(f"Generated data:")
    print(f"  - {len(customers)} customers")
    print(f"  - {len(flight_orders)} orders")
    print(f"  - {len(offer_catalog)} journeys")
    print(f"  - {len(interactions)} interactions")
    print(f"Data saved to data/ directory")


if __name__ == '__main__':
    generate_sample_data()

