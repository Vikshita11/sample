# Hybrid Two-Stage Recommendation System

A production-ready hybrid recommendation system for trip/journey recommendations using:
- **Two-tower embeddings** for user-journey matching
- **FAISS-based ANN** for candidate generation
- **XGBoost re-ranker** for business-aware ranking

## Architecture

1. **Feature Store**: RFM, route affinity, channel preferences
2. **Embedding Model**: Two-tower (user tower + journey tower)
3. **Candidate Generation**: FAISS HNSW index for ANN search
4. **Re-ranking**: XGBoost with vector scores + business features
5. **Serving**: Online pipeline with suppression/cadence rules

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train embeddings
python train_embeddings.py

# Build FAISS index
python build_index.py

# Train re-ranker
python train_reranker.py

# Run evaluation
python evaluate.py

# Serve recommendations
python serve.py --user_id 123 --top_k 10
```

## Project Structure

```
.
├── src/
│   ├── feature_store/      # RFM, route affinity, channel features
│   ├── models/            # Two-tower embedding model
│   ├── retrieval/         # FAISS-based ANN candidate generation
│   ├── reranker/          # XGBoost re-ranker
│   ├── evaluation/        # Metrics (precision@K, recall@K, NDCG)
│   └── serving/           # Online serving pipeline
├── data/                  # Sample data and configs
├── scripts/               # Training and evaluation scripts
└── tests/                 # Unit tests
```

## Roadmap

- [x] Sprint 0: Project setup
- [x] Sprint 1: Feature store
- [x] Sprint 2: Two-tower embeddings
- [x] Sprint 3: FAISS ANN retrieval
- [x] Sprint 4: Re-ranker + evaluation
- [ ] Sprint 5: Production deployment
- [ ] Sprint 6: A/B testing

