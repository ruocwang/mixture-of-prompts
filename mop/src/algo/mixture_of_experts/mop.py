import numpy as np

from algo.utils import generate, evaluate
from algo.mixture_of_experts.mop_trainer import MoPTrainer


def find_prompts(data_man, temp_man, conf):
    args = conf['generation']['args']
    inst_conf = conf['method']['inst']  ## instruction assignment conf

    ## passing pgen_fn to the MoPTrainer to generate prompts from any sample of data
    def pgen_fn(pgen_data, conf, return_score=False, eval=True):
        prompters = generate.generate_prompts(pgen_data, temp_man, conf)
        prompters = list(set(prompters))
        print('Model returns {} unique prompts.'.format(len(prompters)))

        scores = None
        if eval:
            res = evaluate.evaluate_prompts(prompters, data_man, conf['evaluation'], seed=args.seed)
            prompters, scores = res[0].sorted()

        if return_score:
            return prompters, scores
        else:
            return prompters
    
    def eval_fn(prompter):  ## used for RBJS
        return evaluate.evaluate_prompts(prompter,
                                         data_man,
                                         conf['prompt_assignment'], 
                                         seed=args.seed)[1].mean()

    print('Building MoP...')
    mop = MoPTrainer(args.n_experts, conf, temp_man)
    prompter_mop = mop.train_experts(data_man, eval_fn, pgen_fn)

    res = [evaluate.ExecAccuracyEvaluationResult([prompter_mop], np.ones(1))]
    return res
