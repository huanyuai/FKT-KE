#%%
from typing import List, Dict, Any
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset

from editors.fkt_ke import FKTKE


def load_and_format_agnews(split: str, max_samples: int = None) -> List[Dict[str, str]]:
    """加载 AG News 并格式化为分类 prompt 列表。

    返回: [{'prompt': str, 'label': str}, ...]
    """
    ds = load_dataset('ag_news', split=split)
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    samples: List[Dict[str, str]] = []
    total = len(ds) if max_samples is None else min(max_samples, len(ds))
    for i in range(total):
        sample = ds[i]
        prompt = f"Article: {sample['text']}\nCategory:"
        label = label_map[int(sample['label'])]
        samples.append({'prompt': prompt, 'label': label})
    return samples


class FKTKEClient:
    def __init__(self, client_id: int, editor: FKTKE) -> None:
        self.client_id = client_id
        self.editor = editor

    @torch.no_grad()
    def broadcast_prediction(self, prompt: str) -> Dict[str, Any]:
        tok = self.editor.tokenizer
        model = self.editor.model
        device = self.editor.device

        inputs = tok(prompt, return_tensors='pt').to(device)
        outputs = model(**inputs)
        # 取最后一个位置的 logits 作为下一个 token 的分布
        logits = outputs.logits[:, -1, :]  # [1, V]
        probs = F.softmax(logits, dim=-1)  # [1, V]
        conf, pred_id = torch.max(probs, dim=-1)       # [1]
        pred_token_id = int(pred_id[0].item())
        predicted_label = tok.decode([pred_token_id], skip_special_tokens=True)
        return {
            'prompt': prompt,
            'prediction': predicted_label,
            'confidence': float(conf[0].item()),
        }


class FederatedSimulator:
    def __init__(self,
                 clients: List[FKTKEClient],
                 public_dataset: List[Dict[str, str]],
                 test_dataset: List[Dict[str, str]]) -> None:
        self.clients = clients
        self.public_dataset = public_dataset
        self.test_dataset = test_dataset

    def run_round(self) -> None:
        # 1) 抽样一小批公共样本
        if len(self.public_dataset) == 0:
            return
        batch = random.sample(self.public_dataset, k=min(5, len(self.public_dataset)))

        # 2) 广播阶段
        all_broadcasts_for_round: List[Dict[str, Any]] = []
        for sample in batch:
            for client in self.clients:
                pkt = client.broadcast_prediction(sample['prompt'])
                pkt['client_id'] = client.client_id
                all_broadcasts_for_round.append(pkt)

        # 3) 整合阶段
        for client in self.clients:
            knowledge_to_add: List[Dict[str, str]] = []
            metadata_to_add: List[Dict[str, float]] = []
            for sample in batch:
                # 关联到该 sample 的所有广播
                related = [b for b in all_broadcasts_for_round if b['prompt'] == sample['prompt']]
                if not related:
                    continue
                # 当前 client 的预测
                mine_list = [b for b in related if b.get('client_id') == client.client_id]
                if len(mine_list) == 0:
                    continue
                mine = mine_list[0]
                my_pred = mine['prediction']

                # 统计其他客户端的预测（只统计与我不同的冲突知识）
                others = [b for b in related if b.get('client_id') != client.client_id]
                resonance_map: Dict[str, int] = {}
                for b in others:
                    pred = b['prediction']
                    if pred == my_pred:
                        continue
                    resonance_map[pred] = resonance_map.get(pred, 0) + 1

                # 为每个有共振的冲突知识，选择最高置信度代表并记录
                for pred_label, resonance in resonance_map.items():
                    if resonance <= 0:
                        continue
                    cands = [b for b in others if b['prediction'] == pred_label]
                    if not cands:
                        continue
                    rep = max(cands, key=lambda x: x.get('confidence', 0.0))
                    request = {
                        'prompt': sample['prompt'],
                        'target_new': pred_label,
                    }
                    metadata = {
                        'E': float(rep.get('confidence', 0.0)),
                        'R': float(resonance),
                    }
                    knowledge_to_add.append(request)
                    metadata_to_add.append(metadata)

            if len(knowledge_to_add) > 0:
                # 更新当前客户端的本地知识库
                client.editor.edit_batch(knowledge_to_add, metadatas=metadata_to_add)

    def evaluate(self, round_num: int) -> None:
        print(f"\n[Evaluate] Round {round_num}")
        for client in self.clients:
            correct = 0
            total = 0
            for sample in self.test_dataset:
                outs = client.editor.generate_with_memory([sample['prompt']], max_new_tokens=5, do_sample=False)
                text = outs[0] if len(outs) > 0 else ''
                # 取第一个词作为预测标签
                pred = text.strip().split()[:1]
                pred_label = pred[0] if len(pred) > 0 else ''
                if pred_label == sample['label']:
                    correct += 1
                total += 1
            acc = (correct / total) if total > 0 else 0.0
            kb_size = len(getattr(client.editor, 'memory_keys', []))
            print(f"Client {client.client_id} - Acc: {acc:.4f}, KB size: {kb_size}")

    def run(self, num_rounds: int) -> None:
        self.evaluate(round_num=0)
        for i in range(num_rounds):
            print(f"\n[Round] {i+1}/{num_rounds}")
            self.run_round()
            self.evaluate(round_num=i+1)

if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from utils.utils import get_model_editor_config_path
    from editors.fkt_ke import FKTKE, FKTKEConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--edit_model_name', type=str, default='gpt2-xl')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_pub_samples', type=int, default=100)
    parser.add_argument('--max_test_samples', type=int, default=200)
    args = parser.parse_args()

    # 共享底模与分词器
    model_path, config_path = get_model_editor_config_path(args.edit_model_name, 'fkt-ke')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device)

    # 加载配置
    config = FKTKEConfig.from_yaml(config_path)

    # 初始化客户端（共享同一模型/分词器对象）
    clients: List[FKTKEClient] = []
    for cid in range(args.num_clients):
        editor = FKTKE(model, tokenizer, config, args.device)
        clients.append(FKTKEClient(cid, editor))

    # 数据
    public_data = load_and_format_agnews('train', args.max_pub_samples)
    test_data = load_and_format_agnews('test', args.max_test_samples)

    # 模拟器
    sim = FederatedSimulator(clients, public_data, test_data)
    sim.run(args.num_rounds)


