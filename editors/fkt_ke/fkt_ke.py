from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from editors.editor import BaseEditor, EditorConfig
from editors.recipe.models import KnowledgeRepModel, PromptTransformer


@dataclass
class FKTKEConfig(EditorConfig):
    # 基础配置
    edit_model_name: str
    begin_layer_path: str
    lm_head_path: str
    model_hidden_size: int

    # 记忆与检索
    knowledge_rep_dim: int
    knowl_rep_prot_token_n: int
    prompt_token_n: int
    lambda_sim: float
    lambda_E: float
    lambda_R: float
    retr_top_k: int = 1
    retr_min_score: float = -999.0
    # 新增字段
    confidence_threshold: float = 0.7  # 本地模型的自信阈值

    # 组件路径
    krm_base_path: str = 'models/roberta-base'
    # 新增字段：可选的预训练组件权重路径
    krm_ckpt_path: Optional[str] = None
    prompt_transformer_ckpt_path: Optional[str] = None


class FKTKE(BaseEditor):
    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 config: FKTKEConfig,
                 device: str = 'cuda') -> None:
        super().__init__(model, tokenizer, device)
        self.cfg = config
        self.device = device if device != 'auto' else 'cuda:0'

        # 预训练且冻结的知识编码器与提示变换器
        self.knowl_rep_model = KnowledgeRepModel(
            rep_n=config.knowledge_rep_dim,
            prot_token_n=config.knowl_rep_prot_token_n,
            device=self.device,
            base_path=config.krm_base_path,
        )
        # 可选加载任务对齐权重
        if self.cfg.krm_ckpt_path:
            self.knowl_rep_model.load_state_dict(torch.load(self.cfg.krm_ckpt_path, map_location=self.device))
            print(f"Loaded task-aligned KnowledgeRepModel from {self.cfg.krm_ckpt_path}")
        for p in self.knowl_rep_model.parameters():
            p.requires_grad = False

        self.prompt_transformer = PromptTransformer(
            config.knowledge_rep_dim,
            config.model_hidden_size,
            config.prompt_token_n,
            self.device,
        )
        # 可选加载任务对齐权重
        if self.cfg.prompt_transformer_ckpt_path:
            self.prompt_transformer.load_state_dict(torch.load(self.cfg.prompt_transformer_ckpt_path, map_location=self.device))
            print(f"Loaded task-aligned PromptTransformer from {self.cfg.prompt_transformer_ckpt_path}")
        for p in self.prompt_transformer.parameters():
            p.requires_grad = False

        # 内存：键、提示、强度与共振
        self.memory_keys: List[torch.Tensor] = []     # [D]
        self.memory_prompts: List[torch.Tensor] = []  # [T, H]
        self.memory_E: List[float] = []               # 编码强度
        self.memory_R: List[float] = []               # 语义共振

        # 原始模型状态恢复需要
        self._original_forward = self.model.forward

    # --- BaseEditor 接口 ---
    def name_of_editor_and_model(self) -> Tuple[str, str]:
        return 'fkt-ke', self.cfg.edit_model_name

    def if_can_batch_edit(self) -> bool:
        return True

    def edit_one_piece(self, request: Dict):
        # 单条“编辑”即写入一条记忆痕迹（无训练）
        # request 需包含：{'prompt': str, 'target_new': str, 'E': float, 'R': float}
        self.edit_batch([request])

    def edit_batch(self, requests: List[Dict], metadatas: Optional[List[Dict]] = None):
        # 批量写入记忆痕迹
        # 必要字段：prompt/knowledge_text/target_new -> 作为知识句子编码
        for idx, req in enumerate(requests):
            # 构造知识文本（可按需要更精细，这里用 prompt + target_new）
            knowl_text = req.get('knowledge_text', None)
            if knowl_text is None:
                knowl_text = (req.get('prompt', '') + ' ' + req.get('target_new', '')).strip()
            # 键与提示
            with torch.no_grad():
                r_k: torch.Tensor = self._encode_knowledge(knowl_text)  # [D]
                p_k: torch.Tensor = self._value_from_key(r_k)            # [T, H]
            self.memory_keys.append(r_k.detach().to(self.device))
            self.memory_prompts.append(p_k.detach().to(self.device))
            if metadatas is not None and idx < len(metadatas) and metadatas[idx] is not None:
                meta = metadatas[idx]
                self.memory_E.append(float(meta.get('E', 1.0)))
                self.memory_R.append(float(meta.get('R', 0.0)))
            else:
                self.memory_E.append(float(req.get('E', 1.0)))
                self.memory_R.append(float(req.get('R', 0.0)))

    def restore_to_original_model(self):
        # FKT-KE 不修改权重，只需清空记忆或关闭注入
        self.memory_keys.clear()
        self.memory_prompts.clear()
        self.memory_E.clear()
        self.memory_R.clear()

    # --- 推理与注入 ---
    @torch.no_grad()
    def generate_with_memory(self,
                             prompt_texts: List[str],
                             max_new_tokens: int = 5,
                             do_sample: bool = False,
                             arbitration_override: bool = False, # 新增参数
                             **kwargs) -> List[str]:
        
        res: List[str] = []
        for text in prompt_texts:
            # --- 内部仲裁 ---
            use_memory = False
            if arbitration_override:
                # 在 broadcast_prediction 中，如果需要记忆，则强制使用
                best_mem_idx, _ = self.calculate_activation_potential(text)
                if best_mem_idx != -1:
                    use_memory = True
            else:
                # 正常评测流程，执行专家否决仲裁
                self.active_prompt = None
                inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                lora_conf, _ = torch.max(probs, dim=-1)

                if lora_conf.item() < self.cfg.confidence_threshold:
                    best_mem_idx, _ = self.calculate_activation_potential(text)
                    if best_mem_idx != -1:
                        use_memory = True

            # --- 生成 ---
            final_inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            if use_memory:
                best_mem_idx, _ = self.calculate_activation_potential(text) # 再次获取以确保索引正确
                prompt_embeds = self.memory_prompts[best_mem_idx].unsqueeze(0)
                inputs_embeds = self.model.get_input_embeddings()(final_inputs['input_ids'])
                merged_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
                
                # 更新 attention_mask 以包含提示
                mask_to_add = torch.ones(
                    (merged_embeds.size(0), prompt_embeds.size(1)),
                    device=self.device,
                    dtype=final_inputs['attention_mask'].dtype
                )
                new_attention_mask = torch.cat([mask_to_add, final_inputs['attention_mask']], dim=1)

                output_ids = self.model.generate(
                    inputs_embeds=merged_embeds,
                    attention_mask=new_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            else:
                output_ids = self.model.generate(
                    **final_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            input_len = final_inputs['input_ids'].shape[1]
            gen_only_ids = output_ids[0][input_len:]
            res.append(self.tokenizer.decode(gen_only_ids, skip_special_tokens=True))
        return res

    # --- 内部方法 ---
    @torch.no_grad()
    def _encode_knowledge(self, text: str) -> torch.Tensor:
        # 使用冻结的知识编码器得到键向量 r_k
        # 返回形状: [D]
        toks = self.knowl_rep_model.tokenizer(text, return_tensors='pt').to(self.device)
        base_out = self.knowl_rep_model.base_model(**toks).last_hidden_state  # [1, L, 768]
        # 简化：取 cls + pool 等四路拼接已在 RECIPE 中实现；此处复用其 MLP
        # 直接使用 KnowledgeRepModel 的转换器
        # 组成 4*768 的拼接特征（借用 RECIPE 模式：首/尾/avg/max）
        hidden = base_out  # [1, L, 768]
        cls = hidden[:, 0]
        avg = hidden.mean(dim=1)
        mx = hidden.max(dim=1).values
        tail = hidden[:, -1]
        feat = torch.cat([cls, tail, avg, mx], dim=-1)  # [1, 4*768]
        r_k = self.knowl_rep_model.knowl_trans_mlp2(self.knowl_rep_model.knowl_trans_mlp1(feat))  # [1, D]
        return r_k.squeeze(0)

    @torch.no_grad()
    def _value_from_key(self, r_k: torch.Tensor) -> torch.Tensor:
        # 将键向量映射为连续提示向量 [T, H]
        # FIX: Changed from .key_to_prompt(..) to direct call, invoking the forward() method.
        p = self.prompt_transformer(r_k.unsqueeze(0))  # [1, T, H]
        return p.squeeze(0)

    @torch.no_grad()
    def _encode_query(self, text: str) -> torch.Tensor:
        # 复用知识编码器生成查询向量，或若论文区分可在此换另一分支
        return self._encode_knowledge(text)

    @torch.no_grad()
    def calculate_activation_potential(self, query_text: str) -> Tuple[int, float]:
        """
        根据文本查询计算激活潜能，并返回最佳记忆的索引和得分。
        """
        if len(self.memory_keys) == 0:
            return -1, -float('inf')

        # 1) 形成查询向量
        r_q: torch.Tensor = self._encode_query(query_text)

        # 2) 计算激活潜能
        K = torch.stack(self.memory_keys, dim=0)
        r_q_n = r_q / (r_q.norm(p=2) + 1e-8)
        K_n = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (K_n @ r_q_n)
        E = torch.tensor(self.memory_E, device=self.device, dtype=cos.dtype)
        R = torch.tensor(self.memory_R, device=self.device, dtype=cos.dtype)
        A = self.cfg.lambda_sim * cos + self.cfg.lambda_E * E + self.cfg.lambda_R * torch.log1p(R)

        # 3) 找到最佳者
        best_score, best_idx = torch.max(A, dim=0)
        return best_idx.item(), best_score.item()

    @torch.no_grad()
    def _select_memory(self, query_text: str) -> List[int]:
        """
        根据文本查询选择 top-k 的记忆索引。
        """
        if len(self.memory_keys) == 0:
            return []
        best_idx, best_score = self.calculate_activation_potential(query_text)
        if best_score >= self.cfg.retr_min_score:
            return [best_idx]
        return []

