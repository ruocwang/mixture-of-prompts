import openai
import os
import time
import json
import logging
import torch
import random
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pprint import pformat
from misc import TASKS, BBH_TASKS, SUPERNI_TASKS


def make_args():
    parser = argparse.ArgumentParser(description="MoP pipeline")
    #### management
    parser.add_argument('--group', type=str, default='none', help='exps in the same group are all saved to [group] folder')
    parser.add_argument('--save', type=str, default='cmd', help='saving directory / expid')
    parser.add_argument('--tag', type=str, default='none', help='extra tag')

    #### exp setup
    parser.add_argument("--benchmark", type=str, default='ii', help='benchmark name',
                        choices=['ii', 'bbh', 'superni'])
    parser.add_argument("--task", type=str, default=None,
                        help="The name of the dataset to use (via the datasets library).",)
    parser.add_argument("--num_exps", type=int, default=1)
    
    #### search algo
    parser.add_argument("--method", type=str, default='ape', help="Search algo.")
    ## mop
    parser.add_argument("--n_experts", type=int, default=5,
                        help='number of the experts in the mixture')

    args = parser.parse_args()

    if args.benchmark == 'bbh':
        args.task = BBH_TASKS[int(args.task)] if args.task.isdigit() else args.task
    elif args.benchmark == 'superni':
        args.task = SUPERNI_TASKS[int(args.task)] if args.task.isdigit() else args.task
    elif args.benchmark == 'ii':
        args.task = TASKS[int(args.task)] if args.task.isdigit() else args.task
    else:
        raise ValueError(args.benchmark)
    if 'debug' in args.tag: args.group = 'debug'

    ## output dir
    script_name = args.save
    exp_id = '{}'.format(script_name)
    if args.tag and args.tag != 'none': exp_id += f'_tag={args.tag}'
    if 'debug' in args.tag: exp_id = args.tag
    
    args.save = os.path.join('experiments/', f'{args.group}/', exp_id)
    args.log_path = os.path.join(args.save, f'log_{args.task}.txt')
    args.result_path = os.path.join(args.save, f'all_results.json')
    args.prompt_path = os.path.join(args.save, f'all_prompts.json')
    args.eval_score_path = os.path.join(args.save, f'all_eval_scores.json')

    ## create dir
    if not os.path.exists(args.save):
        create_exp_dir(args.save, run_script='./exp_scripts/{}'.format(script_name + '.sh'))
    ## override log fle
    if os.path.exists(args.log_path):
        if 'debug' in args.tag or input(f'{args.log_path} exists, override? [y/n]') == 'y': x=1
        else: exit()
    ## output files
    args.save_plot_path = os.path.join(args.save, 'plots')
    if not os.path.exists(args.save_plot_path): os.mkdir(args.save_plot_path)
    # logging
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(args.log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO) # newly added
    logging.info('\n================== Args ==================\n')
    logging.info(pformat(vars(args)))

    return args


def create_exp_dir(path, run_script=None):
    if not os.path.exists(path):
        os.makedirs(path)
    
    script_path = os.path.join(path, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path)


def pick_gpu_lowest_memory(wait_on_full=0):
    import gpustat

    while True:
        print('queueing for GPU...')
        stats = gpustat.GPUStatCollection.new_query()
        ids = list(map(lambda gpu: int(gpu.entry['index']), stats))
        ratios = list(map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats))
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        print(stats)
        
        if not wait_on_full:
            break

        bestGPUratio = ratios[bestGPU]
        if bestGPUratio < 0.05:
            break
        else:
            time.sleep(1)
    print('found available GPU: {}'.format(bestGPU))
    return bestGPU


def deterministic_mode(seed):
    logging.info('===> Deterministic mode with seed: {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_exp_stats(all_results):
    all_best = []
    all_top10 = []
    for result in all_results:
        all_best.append(result['best_avg_loss'])
        all_top10.append(np.mean(result['top10_avg_losses']))
    
    avg_best, std_best = np.mean(all_best), np.std(all_best)
    avg_top10, std_top10 = np.mean(all_top10), np.std(all_top10)
    
    return (avg_best, std_best), (avg_top10, std_top10)


def save_image(save_image_path, image_pil, prompt, seed, avg_loss, loss, prefix):
    path = f"{save_image_path}/{prefix}-{np.round(abs(avg_loss), 2)}-{prompt}-{seed}-{np.round(abs(loss), 2)}.png"
    image_pil.save(path)
    return path


def plot(points, seed, t, save_plot_path):
    plt.plot(points)
    plt.savefig(f'{save_plot_path}/loss-t={t}-seed={seed}.png')
    plt.close()


def plot_learning_curve(info_list, save_plot_path):
    points = []
    max_score = 0
    for info in info_list:
        points.append(max(abs(info['avg_loss']), max_score))
        max_score = max(points)

    plt.plot(points)
    plt.savefig(f'{save_plot_path}/best_score.png')
    plt.close()


def text_save(lines, file_path):
    with open(file_path, 'w') as f:
        f.writelines([line.strip() + '\n' for line in lines])


def text_load(file_path):
    prompts = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            prompts.append(line.strip())
    return prompts


def json_save(obj, file_path):
    json_object = json.dumps(obj, indent=4)
    json_object = unindent_list(json_object)
    with open(file_path, "w") as outfile:
        outfile.write(json_object)


def json_load(file_path, allow_empty=True):
    if not os.path.exists(file_path) and allow_empty:
        print('loading empty json')
        return {}
    with open(file_path, "r") as outfile:
        obj = json.load(outfile)
    return obj


def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")


def unindent_list(json_object):
    res = ''
    inside_list = False
    idx = 0
    while idx < len(json_object):
        c = json_object[idx]
        if c == '[':
            inside_list = True
        if c == ']':
            inside_list = False

        if inside_list and c == '\n':
            num_spaces = len(json_object[idx + 1:]) - len(json_object[idx + 1:].lstrip())
            idx += num_spaces

        res += json_object[idx]
        idx += 1

    return res


def load_args_from_dict(path):
    args_dict = json_load(path, allow_empty=False)
    return FAKE_ARGS(args_dict)


class FAKE_ARGS:
    def __init__(self, json_dict):
        for key, value in json_dict.items():
            setattr(self, key, value)


class DIRS():
    def __init__(self):
        self.iz_args = 'src/configs/iz_args.json'
        self.benchmarks_dir = 'benchmarks'

directories = DIRS()

