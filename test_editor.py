from __future__ import annotations

import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from editors.recipe.recipe import RECIPEConfig
from fkt_editor import FKTEditor


def _load_dummy_lm(model_name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok


def build_minimum_recipe_config(hidden_size: int = 768) -> RECIPEConfig:
    # A minimal viable RECIPE config. Adjust to your checkpoints if needed.
    return RECIPEConfig(
        prompt_token_n=4,
        edit_model_name="gpt2",
        knowledge_rep_dim=256,
        knowl_rep_prot_token_n=1,
        model_hidden_size=hidden_size,
        begin_layer_path="transformer.h.0.attn",  # adjust if needed
        lm_head_path="lm_head",
        training=RECIPEConfig.TrainingConfig(
            krm_lr=1e-4,
            pt_lr=1e-4,
            relia_lambda=1.0,
            gen_lambda=1.0,
            loc_lambda=1.0,
            contra_lambda=1.0,
            query_knowledge_t=1.0,
            query_prototype_t=1.0,
            constra_hinge_scale=1.0,
            edit_hinge_scale=1.0,
        ),
    )


def test_fkt_editor_ingest_and_activate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_dummy_lm("gpt2")
    model.to(device)
    cfg = build_minimum_recipe_config(hidden_size=model.config.n_embd)

    editor = FKTEditor(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        device=device,
        ckpt_path=None,
    )

    mock_candidates = [
        {"sample": "Paris is the capital of", "label": " France.", "confidence": 0.9, "resonance": 0.6},
        {"sample": "The Eiffel Tower is located in", "label": " Paris.", "confidence": 0.8, "resonance": 0.5},
    ]

    editor.ingest_knowledge(mock_candidates)
    assert len(editor.memory_traces) == len(mock_candidates)

    query_text = "Where is the Eiffel Tower?"
    prompts = editor.dynamic_activation([query_text])
    assert isinstance(prompts, torch.Tensor)
    assert prompts.ndim == 3  # [B, P, H]
    assert prompts.shape[0] == 1
    assert prompts.shape[1] == cfg.prompt_token_n
    assert prompts.shape[2] == cfg.model_hidden_size

    print("FKTEditor ingest/activation test passed.")


if __name__ == "__main__":
    test_fkt_editor_ingest_and_activate()


