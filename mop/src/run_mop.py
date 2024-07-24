import os
import sys
import logging
import time
import numpy as np

sys.path.insert(0, '.')

## Algos
from algo.automatic_prompt_engineer import ape
from algo.mixture_of_experts import mop
from algo.instruct_zero import iz

## LM libraries
from configs.custom.config_zoo import get_conf
from data.data_manager import DataManager
from src.algo.utils import template, evaluate

from misc import set_all_seed
from exp_utils import json_load, json_save, make_args


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    ####### load dataset
    data_man = DataManager(args.benchmark, args.task)

    ## update n_experts if n_experts > n_demos
    args.n_experts = min(data_man.few_shot_data_size, args.n_experts)
    logging.info(f'[INFO] n_demos = {data_man.few_shot_data_size} < n_experts, reducing n_agent size...')


    ######## load configs
    if args.method == 'mop':
        conf = get_conf('mop', args, min(20, len(data_man.eval_data[0])))
    elif args.method in ['ape', 'ape-random', 'ape-kcen']:
        conf = get_conf('ape', args, min(20, len(data_man.eval_data[0])))
    elif args.method in ['iz', 'iz-random', 'iz-kcen']:
        conf = get_conf('iz', args, min(20, len(data_man.eval_data[0])))
    test_conf = get_conf('test', args, min(100, len(data_man.test_data[0])))


    ####### load templates
    temp_man = template.TemplateManager(args.task, args.method)


    ######## main search
    s = time.time()
    all_prompts = []
    all_eval_scores = []
    if args.method == 'mop':
        logging.info('===> Mixture-of-Prompts')

        result = mop.find_prompts(data_man=data_man, temp_man=temp_man, conf=conf)
        result = result[0]
        prompters = result.sorted()[0]

    elif args.method in ['ape', 'ape-random', 'ape-kcen']:
        logging.info('===> Automatic Prompt Engineer')
        result = ape.find_prompts(data_man=data_man, temp_man=temp_man, conf=conf)
        result = result[0]
        all_prompts = [prompter.prompts[0] for prompter in result.sorted()[0]]
        prompters = [result.sorted()[0][0]]
        all_eval_scores = result.sorted()[1]

    elif args.method in ['iz', 'iz-random', 'iz-kcen']:
        logging.info('===> InstructZero')
        result = iz.find_prompts(data_man=data_man, temp_man=temp_man, conf=conf)
        result = result[0]
        all_prompts = [prompter.prompts[0] for prompter in result.sorted()[0]]
        prompters = [result.sorted()[0][0]]

    else:
        raise ValueError(args.method)
    logging.info(f'[INFO] search cost {(time.time() - s) / 60}m')

    ####### testing
    logging.info('Evaluate on test data...')
    
    s = time.time()
    test_res, _, _ = evaluate.evaluate_prompts(prompters[0], data_man, test_conf['evaluation'], seed=args.seed)
    test_score = test_res.sorted()[1][0]
    logging.info(f'[INFO] inference cost {(time.time() - s) / 60}m')

    logging.info("Final Instruction is:")
    logging.info(prompters[0])
    return test_score, all_prompts, all_eval_scores




if __name__ == "__main__":
    args = make_args()


    #### main experiments (repeat for N seeds)
    logging.info('='*20 + f'\n\nRunning task {args.task}\n\n' + '='*20)
    all_prompts = {}
    all_eval_scores = {}
    all_test_scores = []
    for seed in range(args.num_exps):
        args.seed = seed
        logging.info(set_all_seed(args.seed))
        
        test_score, prompts, eval_scores = run(args)
        
        all_prompts[seed] = prompts
        all_eval_scores[seed] = eval_scores
        all_test_scores.append(test_score)
        logging.info(f'Test score on ChatGPT: {test_score}')


    #### save results
    ## prompts
    all_prompts_all_tasks = json_load(args.prompt_path)
    all_prompts_all_tasks[args.task] = all_prompts
    json_save(all_prompts_all_tasks, args.prompt_path)
    ## eval scores
    all_eval_scores_all_tasks = json_load(args.eval_score_path)
    all_eval_scores_all_tasks[args.task] = all_eval_scores
    json_save(all_eval_scores_all_tasks, args.eval_score_path)
    ## scores
    mean, std = np.mean(all_test_scores), np.std(all_test_scores)
    all_results = json_load(args.result_path)
    all_results[args.task] = {
        'TestScore on ChatGPT (avg)': f'{mean:.4f} ({std:.4f})',
        'TestScores': all_test_scores
    }
    json_save(all_results, args.result_path)
    res_str = f'{mean:.4f} ({std:.4f})'
    logging.info('='*20)
    logging.info(f'Results across all runs: {res_str}')
