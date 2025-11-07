# Quick Start Guide

This guide will help you get started with the Hybrid Two-Stage Recommendation System.

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start (End-to-End Example)

Run the complete pipeline example:

```bash
python example_usage.py
```

This will:
1. Generate sample data
2. Compute features (RFM, route affinity, channel preferences)
3. Train the two-tower embedding model
4. Build FAISS index
5. Train re-ranker
6. Serve sample recommendations

## Step-by-Step Workflow

### 1. Generate Sample Data

```bash
python src/data_generator.py
```

This creates sample data in the `data/` directory:
- `customers.csv`
- `flight_orders.csv`
- `offer_catalog.csv`
- `interactions.csv`

### 2. Train Embeddings

```bash
python scripts/train_embeddings.py
```

This trains the two-tower model and saves:
- `models/two_tower_model.pt` - Trained model
- `models/user_embeddings.npy` - User embeddings
- `models/journey_embeddings.npy` - Journey embeddings

### 3. Build FAISS Index

```bash
python scripts/build_index.py
```

This builds the FAISS HNSW index and saves:
- `models/faiss_index.index` - FAISS index
- `models/faiss_index_metadata.pkl` - Metadata mapping

### 4. Train Re-ranker

```bash
python scripts/train_reranker.py
```

This trains the XGBoost re-ranker and saves:
- `models/reranker.model` - Trained re-ranker

### 5. Evaluate

```bash
python scripts/evaluate.py
```

This evaluates the system and prints:
- Precision@K
- Recall@K
- NDCG@K
- MAP@K

### 6. Serve Recommendations

```bash
python scripts/serve.py --user_id 1 --top_k 10
```

Options:
- `--user_id`: User ID to get recommendations for
- `--top_k`: Number of recommendations (default: 10)
- `--filter_region`: Filter by region (e.g., "EU")
- `--filter_channel`: Filter by channel (e.g., "email")

## Project Structure

```
.
├── src/
│   ├── feature_store/      # RFM, route affinity, channel features
│   ├── models/            # Two-tower embedding model
│   ├── retrieval/         # FAISS-based ANN candidate generation
│   ├── reranker/          # XGBoost re-ranker
│   ├── evaluation/         # Metrics (precision@K, recall@K, NDCG)
│   └── serving/           # Online serving pipeline
├── scripts/               # Training and evaluation scripts
├── data/                  # Sample data (generated)
├── models/                # Trained models and indexes (generated)
├── config.yaml            # Configuration file
└── example_usage.py       # Complete end-to-end example
```

## Configuration

Edit `config.yaml` to customize:
- Model parameters (embedding_dim, hidden_dims, etc.)
- FAISS index type (HNSW, IVF, IVF_HNSW)
- Retrieval parameters (candidate_count, top_k)
- Re-ranker parameters (n_estimators, max_depth, etc.)

## Key Features

1. **Two-Tower Embeddings**: User and journey embeddings for semantic matching
2. **FAISS ANN Search**: Fast approximate nearest neighbor search (HNSW index)
3. **XGBoost Re-ranker**: Combines vector scores with business features
4. **Feature Store**: RFM segmentation, route affinity, channel preferences
5. **Suppression Rules**: Consent and cadence enforcement
6. **Evaluation Metrics**: Precision@K, Recall@K, NDCG@K, MAP@K

## Next Steps

- Replace sample data with your real data
- Tune hyperparameters in `config.yaml`
- Add custom business rules in `src/serving/pipeline.py`
- Scale to production with Milvus/Pinecone (see `info.txt`)

## Troubleshooting

**Import errors**: Make sure you're running scripts from the project root directory.

**FAISS errors**: Install `faiss-cpu` or `faiss-gpu` depending on your system.

**Model not found**: Run training scripts first to generate models.

**Data not found**: Run `python src/data_generator.py` to generate sample data.

