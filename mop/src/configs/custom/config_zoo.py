from algo.utils import config
from .config_algo import get_algo_conf


def mop(args, samples_per_eval):
    conf = {
        'generation': {
            'num_subsamples': 20 // args.n_experts, # reduced for fair comparison with APE
            'num_demos': 5,  # reduced because MoP receives less total demos during prompt generation
            'num_prompts_per_subsample': 1,
        },
        'evaluation': {
            'task': args.task,
            'num_samples': samples_per_eval,
            'threshold': True,
        },
        'prompt_assignment': {
            'task': args.task,
            'num_samples': samples_per_eval * args.n_experts,
        }
    }

    return conf


def ape(args, samples_per_eval):
    conf = {
        'generation': {
            'num_subsamples': 20,
            'num_demos': 10,
            'num_prompts_per_subsample': 1,  ## 1 since the seed is fixed for each LLM call
        },
        'evaluation': {
            'task': args.task,
            'num_samples': samples_per_eval,
        }
    }

    return conf


def iz(args, samples_per_eval):
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 10,
            'num_prompts_per_subsample': 20,
        },
        'evaluation': {
            'task': args.task,
            'num_samples': samples_per_eval,
        }
    }

    return conf


def test(args, num_test_samples):
    test_conf = {
        'evaluation': {
            'task': args.task,
            'num_samples': num_test_samples,
            'split': 'test',
        }
    }

    return test_conf
    

def get_conf(conf, args, samplers_per_eval):
    if conf == 'mop':
        conf = mop(args, samplers_per_eval)
    elif conf == 'ape':
        conf = ape(args, samplers_per_eval)
    elif conf == 'iz':
        conf = iz(args, samplers_per_eval)
    elif conf == 'test':
        conf = test(args, samplers_per_eval)
    else:
        raise ValueError(f'Unknown conf: {conf}')
    conf = update_api_model_conf(conf, args.seed)
    conf = update_api_args_conf(conf, args)

    base_conf='../../configs/base/llm_base_conf.yaml'
    conf = config.update_config(conf, base_conf)

    ## load search algo parameters
    if args.method == 'mop':
        algo_conf = get_algo_conf('kauto_rbjs')
    elif args.method in ['ape', 'iz']:
        algo_conf = get_algo_conf('none')
    elif args.method in ['ape-random', 'iz-random']:
        algo_conf = get_algo_conf('greedy_fix_icl')
    elif args.method in ['ape-kcen', 'iz-kcen']:
        algo_conf = get_algo_conf('greedy_kcen_icl')
    conf['method'] = algo_conf

    return conf


def update_api_model_conf(conf, seed):
    model_conf = {
            "name": "GPT_forward",
            'gpt_config': {
                'model': 'gpt-3.5-turbo-instruct-0914',
                'seed': seed,
            }
        }

    for key in conf:
        conf[key]['model'] = model_conf
    
    return conf


def update_api_args_conf(conf, args):
    for key in conf:
        conf[key]['args'] = args
    
    return conf
