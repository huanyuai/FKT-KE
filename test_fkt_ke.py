#%%
import os, argparse, sys
from utils.utils import get_editor
from evaluation.editor_eval import EditorEvaluation
from utils.data import TestSampleList


def eval_multi_edit(editor, eval_data_name = 'ZSRE', data_path = None,
        edit_type:str = 'sequential', edit_n:int = 10, data_sample_n:int = None,
        shuffle = True, seed = 0, extra_evaluation_name = None):
    data_map = {
        'zsre': ['data/evaluation/zsre/zsre_mend_eval.json', TestSampleList.zsre],
        'cf': ['data/evaluation/cf/counterfact-edit.json', TestSampleList.counterfact],
        'ripe': ['data/evaluation/ripple_effect/ripe_test.json', TestSampleList.ripple_effect]
    }
    eval_data_name = eval_data_name.lower()
    assert eval_data_name in data_map.keys()
    data_path = data_map[eval_data_name][0] if data_path == None else data_path
    test_sample_list = data_map[eval_data_name][1](data_path, data_sample_n, shuffle, seed)
    evaluation_name = eval_data_name.upper()
    if extra_evaluation_name != None:
        evaluation_name += '-' + extra_evaluation_name
    ev = EditorEvaluation(editor, test_sample_list, evaluation_name)
    if edit_n == 1:
        ev.evaluate_single_edit()
    else:
        if edit_type == 'sequential':
            ev.evaluate_sequential_edit(edit_n, True, seed)
        elif edit_type == 'batch':
            ev.evaluate_batch_edit(edit_n, True, seed)
        else:
            raise


def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edit_model_name', type=str, default='gpt2-xl')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data_name', type=str, default='ZSRE')
    parser.add_argument('--edit_type', type=str, default='sequential')
    parser.add_argument('--edit_n', type=int, default=10)
    parser.add_argument('--data_sample_n', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    cfg = get_attr()
    editor = get_editor('fkt-ke', cfg.edit_model_name, cfg.device)
    eval_multi_edit(editor, cfg.data_name, None, cfg.edit_type, cfg.edit_n, cfg.data_sample_n, True, cfg.seed, None)


