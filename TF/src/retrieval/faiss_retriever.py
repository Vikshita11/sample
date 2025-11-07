"""
FAISS-based ANN Candidate Generation
HNSW index for high-recall retrieval
"""

import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSRetriever:
    """FAISS-based ANN retriever for candidate generation"""
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 index_type: str = "HNSW",
                 hnsw_m: int = 32,
                 hnsw_ef_construction: int = 200,
                 hnsw_ef_search: int = 50,
                 nprobe: int = 10):
        """
        Initialize FAISS retriever
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: "HNSW", "IVF", or "IVF_HNSW"
            hnsw_m: HNSW parameter M (number of connections)
            hnsw_ef_construction: HNSW ef_construction
            hnsw_ef_search: HNSW ef_search
            nprobe: IVF nprobe parameter
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.nprobe = nprobe
        
        self.index = None
        self.journey_ids = None  # Map index position to journey_id
        self.metadata = None  # Journey metadata for filtering
    
    def build_index(self, 
                   journey_embeddings: np.ndarray,
                   journey_ids: List[int],
                   metadata: Optional[Dict] = None):
        """
        Build FAISS index from journey embeddings
        
        Args:
            journey_embeddings: (num_journeys, embedding_dim) numpy array
            journey_ids: List of journey IDs corresponding to embeddings
            metadata: Optional dict mapping journey_id to metadata (for filtering)
        """
        logger.info(f"Building {self.index_type} index for {len(journey_embeddings)} journeys...")
        
        num_journeys, dim = journey_embeddings.shape
        assert dim == self.embedding_dim, f"Embedding dim mismatch: {dim} != {self.embedding_dim}"
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(journey_embeddings)
        
        # Create index based on type
        if self.index_type == "HNSW":
            # HNSW index for high recall
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            self.index.hnsw.efConstruction = self.hnsw_ef_construction
            self.index.hnsw.efSearch = self.hnsw_ef_search
        elif self.index_type == "IVF":
            # IVF index for memory efficiency
            nlist = min(100, num_journeys // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.nprobe = self.nprobe
        elif self.index_type == "IVF_HNSW":
            # Hybrid IVF + HNSW
            nlist = min(100, num_journeys // 10)
            quantizer = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.nprobe = self.nprobe
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Train index if needed (IVF requires training)
        if self.index_type.startswith("IVF"):
            logger.info("Training IVF index...")
            self.index.train(journey_embeddings)
        
        # Add embeddings to index
        self.index.add(journey_embeddings)
        
        # Store journey IDs mapping
        self.journey_ids = np.array(journey_ids)
        self.metadata = metadata or {}
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(self,
              user_embedding: np.ndarray,
              top_k: int = 1000,
              filter_dict: Optional[Dict] = None) -> Tuple[List[int], List[float]]:
        """
        Search for top-k similar journeys
        
        Args:
            user_embedding: (embedding_dim,) numpy array
            top_k: Number of candidates to retrieve
            filter_dict: Optional dict for metadata filtering (e.g., {'region': 'EU', 'consent': True})
        
        Returns:
            journey_ids: List of journey IDs
            scores: List of similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize user embedding
        user_embedding = user_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(user_embedding)
        
        # Set search parameters
        if self.index_type == "HNSW":
            self.index.hnsw.efSearch = self.hnsw_ef_search
        elif self.index_type.startswith("IVF"):
            self.index.nprobe = self.nprobe
        
        # Search (retrieve more if filtering)
        search_k = top_k * 10 if filter_dict else top_k
        distances, indices = self.index.search(user_embedding, min(search_k, self.index.ntotal))
        
        # Map indices to journey IDs
        candidate_indices = indices[0]
        candidate_scores = 1 - distances[0]  # Convert distance to similarity (cosine)
        
        # Apply metadata filtering if provided
        if filter_dict:
            filtered_journey_ids = []
            filtered_scores = []
            
            for idx, score in zip(candidate_indices, candidate_scores):
                journey_id = int(self.journey_ids[idx])
                
                # Check metadata filters
                if self._matches_filter(journey_id, filter_dict):
                    filtered_journey_ids.append(journey_id)
                    filtered_scores.append(float(score))
                    
                    if len(filtered_journey_ids) >= top_k:
                        break
            
            return filtered_journey_ids, filtered_scores
        
        # Return top-k
        journey_ids = [int(self.journey_ids[idx]) for idx in candidate_indices[:top_k]]
        scores = [float(score) for score in candidate_scores[:top_k]]
        
        return journey_ids, scores
    
    def _matches_filter(self, journey_id: int, filter_dict: Dict) -> bool:
        """Check if journey matches filter criteria"""
        if journey_id not in self.metadata:
            return True  # No metadata = pass filter
        
        journey_meta = self.metadata[journey_id]
        
        for key, value in filter_dict.items():
            if key not in journey_meta:
                continue
            if journey_meta[key] != value:
                return False
        
        return True
    
    def save(self, filepath: str):
        """Save index to file"""
        if self.index is None:
            raise ValueError("No index to save")
        
        logger.info(f"Saving index to {filepath}...")
        faiss.write_index(self.index, filepath)
        
        # Save metadata
        metadata_file = filepath.replace('.index', '_metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'journey_ids': self.journey_ids,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        logger.info("Index saved successfully")
    
    def load(self, filepath: str):
        """Load index from file"""
        logger.info(f"Loading index from {filepath}...")
        self.index = faiss.read_index(filepath)
        
        # Load metadata
        metadata_file = filepath.replace('.index', '_metadata.pkl')
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.journey_ids = data['journey_ids']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data['index_type']
        
        logger.info(f"Index loaded with {self.index.ntotal} vectors")

