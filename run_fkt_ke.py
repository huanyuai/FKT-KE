#%%
import os, argparse
from typing import List

from utils.utils import get_editor


def run_demo(num_clients: int = 2,
             edit_model_name: str = 'gpt2-xl',
             device: str = 'auto'):
    # 初始化多个客户端（编辑器实例）
    clients = [get_editor('fkt-ke', edit_model_name, device) for _ in range(num_clients)]

    # 演示：客户端0 发现一条知识并广播给其他客户端
    discovered = {
        'prompt': 'The capital of France is',
        'target_new': ' Paris',
        'E': 1.0,   # 编码强度
        'R': 2.0    # 语义共振（示例）
    }
    clients[0].edit_one_piece(discovered)
    for i in range(1, num_clients):
        clients[i].edit_one_piece(discovered)

    # 测试生成
    texts = ['The capital of France is']
    for i, cli in enumerate(clients):
        out = cli.generate_with_memory(texts, max_new_tokens=12, do_sample=False)
        print(f'[Client {i}] => {out[0]}')


def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--edit_model_name', type=str, default='gpt2-xl')
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


if __name__ == '__main__':
    cfg = get_attr()
    run_demo(cfg.num_clients, cfg.edit_model_name, cfg.device)


