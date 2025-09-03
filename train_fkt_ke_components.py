#%%
import os
import argparse
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from editors.recipe.recipe import RECIPE, RECIPEConfig
from utils.utils import get_model_editor_config_path, model_path_map, set_tokenizer_pad_id
from utils.data import prompts_last_len_to_x_y_mask
from run_federated_classification import load_and_format_agnews


def build_get_data_by_ids(tokenizer: AutoTokenizer,
                          samples: List[Dict[str, str]],
                          device: str) -> Tuple[int, callable]:
    """
    构造与 RECIPE 训练循环兼容的 get_data_by_ids 函数，仅关注可靠性损失。
    返回: (sample_count, get_data_by_ids)
    """
    # 预提取文本，避免多次索引
    prompts = [s['prompt'] for s in samples]
    labels = [s['label'] for s in samples]
    knowl_texts = [f"{p} {l}" for p, l in zip(prompts, labels)]

    def stack_xym(x_ids, y_ids, masks):
        # inputs 已计算为批量张量，直接返回三元组
        return x_ids, y_ids, masks

    def get_data_by_ids(ids: List[int]):
        sel_prompts = [prompts[i] for i in ids]
        sel_labels = [labels[i] for i in ids]
        sel_knowl = [knowl_texts[i] for i in ids]

        # 可靠性：基于 knowledge 文本，让最后一段为监督目标
        # 这里选择 pre_len=0.9，表示前 90% 作为条件、后 10% 作为目标（近似）
        x_ids, y_ids, masks = prompts_last_len_to_x_y_mask(
            tokenizer, sel_knowl, pre_len=0.9, truncation=256, device=device
        )
        batch_relia_xym = stack_xym(x_ids, y_ids, masks)

        # 为兼容结构，gen/loc 复用可靠性批（训练时将其 loss 权重置为 0）
        batch_gen_xym = {'original': batch_relia_xym}
        batch_loc_xym = {'original_loc': batch_relia_xym}

        contra_knowl = sel_knowl
        contra_q_rel = sel_prompts
        contra_q_gen = sel_prompts
        contra_q_loc = {'original_loc': sel_prompts}

        contra_data = (contra_knowl, contra_q_rel, contra_q_gen, contra_q_loc)
        edit_data = (batch_relia_xym, batch_gen_xym, batch_loc_xym)
        return contra_data, edit_data

    return len(samples), get_data_by_ids


def train_components(model_name: str,
                     device: str = 'cuda',
                     epochs: int = 1,
                     batch_size: int = 8,
                     max_train_samples: int = 500,
                     save_dir: str = None):
    # 路径与配置
    model_path, config_path = get_model_editor_config_path(model_name, 'recipe')

    # 模型/分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    set_tokenizer_pad_id(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

    # 配置与权重调整（仅训练可靠性）
    config: RECIPEConfig = RECIPEConfig.from_yaml(config_path)
    config.training.gen_lambda = 0.0
    config.training.loc_lambda = 0.0
    config.training.contra_lambda = 0.0

    # 编辑器与数据
    editor = RECIPE(model, tokenizer, config, device, model_path_map['roberta-base'])
    train_samples = load_and_format_agnews('train', max_train_samples)
    sample_count, get_data_by_ids = build_get_data_by_ids(tokenizer, train_samples, device)

    # 训练初始化与开始
    records_dir = os.path.join('train_records', 'fkt_ke_components')
    os.makedirs(records_dir, exist_ok=True)
    editor.train_init(sample_count, get_data_by_ids,
                      batch_size=batch_size,
                      records_dir=records_dir,
                      train_name_prefix='FKTKE-COMP',
                      train_name=None,
                      load_ckpt_path=None,
                      save_ckpt_per_i=1000000,  # 禁用中途保存
                      log_per_i=10,
                      random_seed=1)
    editor.train(epochs)

    # 保存组件权重
    out_dir = save_dir if save_dir else os.path.join('trained_components', model_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(editor.knowl_rep_model.state_dict(), os.path.join(out_dir, 'knowl_rep_model.pt'))
    torch.save(editor.prompt_transformer.state_dict(), os.path.join(out_dir, 'prompt_transformer.pt'))
    print(f"Saved components to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-xl')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_train_samples', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    # 解析设备
    if args.device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        resolved_device = args.device
    print(f"Resolved device: {resolved_device}")

    train_components(args.model_name,
                     device=resolved_device,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     max_train_samples=args.max_train_samples,
                     save_dir=args.save_dir)


