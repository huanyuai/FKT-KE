from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F

from editors.recipe.recipe import RECIPE


class FKTEditor(RECIPE):
    """
    Federated Knowledge Traces Editor

    Extends RECIPE by replacing the static knowledge/prompt bases with dynamic
    memory traces and providing ingestion and activation mechanisms.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Remove original bases; use dynamic memory traces instead
        if hasattr(self, 'knowledge_base'):
            delattr(self, 'knowledge_base')
        if hasattr(self, 'prompts_base'):
            delattr(self, 'prompts_base')
        self.memory_traces: List[Dict[str, Any]] = []

        # Editors should not update encoders during ingest/activate by default
        self.knowl_rep_model.eval()
        self.prompt_transformer.eval()
        for p in self.knowl_rep_model.parameters():
            p.requires_grad_(False)
        for p in self.prompt_transformer.parameters():
            p.requires_grad_(False)

    # --------------------------- Knowledge Ingestion --------------------------- #
    @torch.no_grad()
    def ingest_knowledge(self, candidates: List[Dict[str, Any]]) -> None:
        """
        candidates: List[{
            'sample': str,
            'label': str | int,
            'confidence': float,          # encoding_strength
            'resonance': float            # semantic_resonance
        }]
        """
        if not candidates:
            return

        # Compose textual knowledge representation
        texts: List[str] = []
        for c in candidates:
            sample = str(c.get('sample', ''))
            label = str(c.get('label', ''))
            if sample and label:
                if sample[-1] != ' ' and label[0] != ' ':
                    texts.append(sample + ' ' + label)
                else:
                    texts.append(sample + label)
            else:
                texts.append(sample or label)

        # Encode knowledge to key vectors and transform to prompt vectors
        key_vectors: torch.Tensor = self.knowl_rep_model(texts, knowl_or_query='k')
        prompt_vectors: torch.Tensor = self.prompt_transformer(key_vectors)

        # Store per-trace dicts
        for i, c in enumerate(candidates):
            trace = {
                'key_vector': key_vectors[i].detach().clone(),
                'prompt_vector': prompt_vectors[i].detach().clone(),
                'encoding_strength': float(c.get('confidence', 1.0)),
                'semantic_resonance': float(c.get('resonance', 0.0)),
                'meta': {
                    'sample': c.get('sample'),
                    'label': c.get('label'),
                }
            }
            self.memory_traces.append(trace)

    # --------------------------- Dynamic Activation --------------------------- #
    @torch.no_grad()
    def dynamic_activation(self, query_texts: List[str],
                           lambda_sim: float = 1.0,
                           lambda_E: float = 1.0,
                           lambda_R: float = 1.0) -> torch.Tensor:
        """
        Given queries, compute activation potential over memory traces and
        select the best prompt vector for each query.

        Returns: Tensor [batch, prompt_token_n, model_hidden_size]
        If no memory traces exist, returns zeros of appropriate prompt shape.
        """
        device = self.device if hasattr(self, 'device') else 'cpu'

        # If empty memory, return zeros
        if len(self.memory_traces) == 0:
            z = torch.zeros([1, self.cfg.prompt_token_n, self.cfg.model_hidden_size], device=device)
            return z.repeat(len(query_texts), 1, 1)

        # Encode queries
        query_vectors: torch.Tensor = self.knowl_rep_model(query_texts, knowl_or_query='q')  # [B, D]

        # Stack all key vectors and prompt vectors from memory
        all_key_vectors = torch.stack([t['key_vector'] for t in self.memory_traces], dim=0).to(device)  # [N, D]
        all_prompt_vectors = torch.stack([t['prompt_vector'] for t in self.memory_traces], dim=0).to(device)  # [N, P, H]

        # Similarity (cosine) scaled; use normalized vectors
        q_norm = F.normalize(query_vectors.to(device), p=2, dim=1)
        k_norm = F.normalize(all_key_vectors, p=2, dim=1)
        sim = q_norm @ k_norm.T  # [B, N]

        # Strength and resonance tensors
        strengths = torch.tensor([t['encoding_strength'] for t in self.memory_traces], device=device).unsqueeze(0)  # [1, N]
        resonances = torch.tensor([t['semantic_resonance'] for t in self.memory_traces], device=device).unsqueeze(0)  # [1, N]

        # Activation potential per paper Eq.(4): weighted sum of components
        activation = lambda_sim * sim + lambda_E * strengths + lambda_R * resonances  # [B, N]

        # Select best trace per query
        best_idx = activation.argmax(dim=1)  # [B]
        selected_prompts = all_prompt_vectors[best_idx]  # [B, P, H]
        return selected_prompts


