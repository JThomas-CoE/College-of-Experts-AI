"""
Semantic Router - V11 Semantic Routing Engine

⚠️  DEPRECATION WARNING ⚠️
This module is DEPRECATED as of 2026-02-01.
For new development, use:
- `MemoryVectorRouter` from `src.memory_router` for memory-aware routing
- `VectorRouter` from `src.vector_router` for FAISS-based similarity search

Kept for backward compatibility with:
- src/harness_v11.py
- demo_v12_e2e.py
"""

import os
import torch
import json
from typing import List, Dict, Tuple

# PATCH: Fix for custom AMD PyTorch builds missing distributed.is_initialized
# This is required for sentence_transformers to run on this specific Ryzen AI setup
if not hasattr(torch, 'distributed'):
    class MockDistributed:
        def is_initialized(self): return False
        def get_rank(self): return 0
        def get_world_size(self): return 1
    torch.distributed = MockDistributed()
elif not hasattr(torch.distributed, 'is_initialized'):
    torch.distributed.is_initialized = lambda: False

import numpy as np
from sentence_transformers import SentenceTransformer, util
from .expert_catalog import ExpertCatalog

class SemanticRouter:
    """
    V11 Semantic Routing Engine.
    Uses Vector Embeddings (BAAI/bge-m3) to match Tasks to Experts.
    """
    
    def __init__(self, catalog: ExpertCatalog, model_name: str = 'BAAI/bge-m3'):
        print(f"[SemanticRouter] Initializing Embedding Engine ({model_name})...")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[SemanticRouter] Warning: Failed to load {model_name}. Fallback to 'all-MiniLM-L6-v2'")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.catalog = catalog
        self.expert_ids = []
        self.expert_embeddings = None
        
        self._index_experts()
        
    def _index_experts(self):
        """Encode all expert profiles into the vector space."""
        print("[SemanticRouter] Indexing Expert Catalog...")
        descriptions = []
        self.expert_ids = []
        
        for expert_id, expert_def in self.catalog.experts.items():
            # SKIP reasoners to prevent vector interference with planning models
            if "reasoner" in expert_id:
                continue
                
            # Construct a rich semantic profile
            caps_str = ", ".join(expert_def.capabilities)
            profile = f"FUNCTIONAL CAPABILITIES: {caps_str}. SYSTEM ROLE: {expert_def.name}. LOGIC: {expert_def.system_prompt}"
            descriptions.append(profile)
            self.expert_ids.append(expert_id)
            
        # Compute embeddings in batch
        self.expert_embeddings = self.model.encode(descriptions, convert_to_tensor=True)
        print(f"[SemanticRouter] Indexed {len(self.expert_ids)} experts.")

    def route_task(self, task_description: str) -> Tuple[str, float]:
        """
        Find the best expert for a single task chunk.
        Returns: (expert_id, confidence_score)
        """
        task_emb = self.model.encode(task_description, convert_to_tensor=True)
        
        # Cosine Similarity
        scores = util.cos_sim(task_emb, self.expert_embeddings)[0]
        
        # Find Winner
        best_idx = scores.argmax().item()
        best_id = self.expert_ids[best_idx]
        best_score = scores[best_idx].item()
        
        return best_id, best_score

    def analyze_plan_clarity(self, chunks: List[Dict], threshold: float = 0.15) -> List[Dict]:
        """
        Calculates 'Semantic Sharpness' for each chunk.
        - High Sharpness = Large gap between Top 1 and Top 2 (Contrast).
        - Low Sharpness = Muddy chunk (confuses the router).
        """
        if self.expert_embeddings is None:
            self._index_experts()
            
        reports = []
        print(f"\n[SemanticRouter] Analyzing Plan Clarity (Contrast Threshold: {threshold})...")
        
        for chunk in chunks:
            # Type safety check
            if isinstance(chunk, str):
                task_desc = chunk
                chunk_id = "unknown"
            else:
                task_desc = chunk.get("description", "")
                chunk_id = chunk.get("id", "c?")
            
            # Get normalized embeddings
            task_emb = self.model.encode([task_desc], convert_to_tensor=True)
            scores = util.cos_sim(task_emb, self.expert_embeddings)[0]
            
            # Get Top 2
            top_vals, top_indices = scores.topk(2)
            
            s1 = top_vals[0].item()
            s2 = top_vals[1].item()
            contrast = s1 - s2
            
            e1 = self.expert_ids[top_indices[0]]
            e2 = self.expert_ids[top_indices[1]]
            
            is_muddy = contrast < threshold
            
            report = {
                "id": chunk_id,
                "description": task_desc,
                "top_expert": e1,
                "top_score": s1,
                "runner_up": e2,
                "runner_up_score": s2,
                "contrast": contrast,
                "is_muddy": is_muddy
            }
            reports.append(report)
            
            status = "[MUDDY]" if is_muddy else "[SHARP]"
            print(f"  Chunk {report['id']}: {status} (Contrast: {contrast:.4f}, {e1} vs {e2})")
            
        return reports

    def optimize_team_assignment(self, chunks: List[Dict]) -> List[Dict]:
        """
        Global Optimization: Assigns experts to maximize the total 'Team Score' (sum of cosine similarities).
        Prevents expert monopolies and ensures the globally best collaboration.
        Uses the Hungarian Algorithm (linear_sum_assignment).
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            print("[SemanticRouter] Warning: scipy not found. Falling back to greedy assignment.")
            return self.assign_experts(chunks)

        if self.expert_embeddings is None:
            self._index_experts()

        num_chunks = len(chunks)
        num_experts = len(self.expert_ids)
        
        # Create Cost Matrix (Scipy minimizes, so we use 1 - score)
        # Dimensions: [Chunks, Experts]
        cost_matrix = np.zeros((num_chunks, num_experts))
        
        print(f"\n[SemanticRouter] Running Global Team Optimizer (Matrix: {num_chunks}x{num_experts})...")
        
        for i, chunk in enumerate(chunks):
            task_desc = chunk.get("description", "")
            task_emb = self.model.encode([task_desc], convert_to_tensor=True)
            scores = util.cos_sim(task_emb, self.expert_embeddings)[0]
            
            for j in range(num_experts):
                # Scipy minimizes cost, so higher similarity = lower cost
                cost_matrix[i, j] = 1.0 - scores[j].item()

        # Solve the Assignment Problem
        # row_ind = chunk indices, col_ind = expert indices
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Apply Assignments
        for i, expert_idx in zip(row_ind, col_ind):
            chunk = chunks[i]
            expert_id = self.expert_ids[expert_idx]
            score = 1.0 - cost_matrix[i, expert_idx]
            
            print(f"  Assigning Chunk {chunk['id']} -> {expert_id} (Score: {score:.4f})")
            chunk["assigned_expert"] = expert_id
            chunk["routing_score"] = score
            
        return chunks

    def assign_experts(self, chunks: List[Dict]) -> List[Dict]:
        """Legacy Greedy Assignment (Fallback or Single-Chunk use)"""
        print("\n[SemanticRouter] Reviewing Assignments for Diversity...")
        for chunk in chunks:
            original_expert = chunk.get("assigned_expert", "unknown")
            task_desc = chunk.get("description", "")
            best_id, score = self.route_task(task_desc)
            
            final_expert = original_expert
            if score > 0.45 and best_id != original_expert:
                final_expert = best_id
            
            chunk["assigned_expert"] = final_expert
            chunk["routing_score"] = score
            
        return chunks
