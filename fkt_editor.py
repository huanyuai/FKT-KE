from __future__ import annotations

from typing import List, Dict, Any

import torch
import torch.nn.functional as F

from editors.recipe.models import KnowledgeRepModel, PromptTransformer
from editors.recipe.recipe import RECIPEConfig


class FKTEditor:
    """
    Federated Knowledge Traces Editor (standalone)

    - Holds a reference to the client classification model (e.g., BERT)
    - Maintains dynamic memory traces (key/prompt vectors)
    - Injects prompts into BERT by replacing the first prompt_token_n
      embeddings inside word_embeddings output (requires caller to prepend
      pad tokens and attention mask ones externally)
    """

    def __init__(self, client_model: torch.nn.Module, config: RECIPEConfig, device: str = 'cuda') -> None:
        self.model = client_model
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device

        # Knowledge modules
        self.knowl_rep_model = KnowledgeRepModel(
            rep_n=config.knowledge_rep_dim,
            prot_token_n=config.knowl_rep_prot_token_n,
            device=self.device,
            base_path='models/roberta-base',
        )
        self.prompt_transformer = PromptTransformer(
            in_dim=config.knowledge_rep_dim,
            out_dim=config.model_hidden_size,
            prompt_token_n=config.prompt_token_n,
            device=self.device,
        )

        self.memory_traces: List[Dict[str, Any]] = []
        self.adopted_prompts: List[torch.Tensor] | torch.Tensor | None = None

        # Freeze auxiliary modules by default
        self.knowl_rep_model.eval()
        self.prompt_transformer.eval()
        for p in self.knowl_rep_model.parameters():
            p.requires_grad_(False)
        for p in self.prompt_transformer.parameters():
            p.requires_grad_(False)

        # Register BERT embedding hook if available
        self.register_bert_hook()

    # --------------------------- Hook into BERT --------------------------- #
    def register_bert_hook(self) -> None:
        try:
            embeddings = self.model.bert.embeddings
            word_embeddings = embeddings.word_embeddings
        except Exception:
            # Model is not a BERT-like architecture; skip hook
            return

        def forward_hook(module, args, output):
            # output: [batch, seq_len, hidden]
            if self.adopted_prompts is None:
                return output

            if isinstance(self.adopted_prompts, list):
                prompts = torch.stack(self.adopted_prompts, dim=0)
            else:
                prompts = self.adopted_prompts
            # prompts: [batch, prompt_len, hidden]
            if prompts is None or prompts.numel() == 0:
                return output

            bsz, seq_len, hidden = output.shape
            p_b, p_len, p_h = prompts.shape
            if p_b != bsz or p_h != hidden:
                return output

            # Replace the first p_len token embeddings with prompts
            if p_len > seq_len:
                # If caller didn't extend input_ids, we cannot safely inject; skip
                return output
            output = output.clone()
            output[:, :p_len, :] = prompts.to(output.device)
            return output

        # Register hook
        word_embeddings.register_forward_hook(forward_hook)

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
        # Cache for hook consumption
        self.adopted_prompts = selected_prompts
        return selected_prompts


