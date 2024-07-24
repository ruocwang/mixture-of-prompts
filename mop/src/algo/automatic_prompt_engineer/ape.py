import random
import numpy as np
from src.algo.utils import generate, evaluate
from src.pe_manager import Prompter
from algo.prompt_lib import load_pregen_prompts
from .demos_allocator import DemosAllocater

def find_prompts(data_man, temp_man, conf):
    args = conf['generation']['args']

    # Generate prompts
    prompters = generate.generate_prompts(data_man.prompt_gen_data, temp_man, conf)
    prompters = list(set(prompters))
    print('Model returns {} unique prompts.'.format(len(prompters)))

    ## evaluate
    res = evaluate.evaluate_prompts(prompters, data_man, conf['evaluation'], seed=args.seed)
    prompters = res[0].sorted()[0]

    ## allocate demos
    demos_allocater = DemosAllocater(args.n_experts, conf, temp_man)
    prompters = demos_allocater.allocate_demos(data_man, prompters)

    ## prepare return
    res = [evaluate.ExecAccuracyEvaluationResult(prompters, np.ones(1))]
    return res
