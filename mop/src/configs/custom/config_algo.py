from algo.utils import config


def none():  ## default for ape
    conf = {
        'demo': {
            'algo': 'none',
        }
    }
    return conf

def greedy_fix_icl():  ## icl (random but fixed) for ape, a.k.a., ape-random
    conf = {
        'demo': {
            'encoder': 'ada',
            'algo': 'fix',
        }
    }
    return conf

def greedy_kcen_icl():  ## icl + kcentroids
    conf = {
        'demo': {
            'encoder': 'ada',
            'algo': 'kcen',
        }
    }
    return conf


def kauto_rbjs():
    conf = {
        'inst': {
            'pgen': 'compensate',
            'cand': 'shared', # all regions share the same candidate prompts
            'algo': 'rbjs',
            'n_inits': 4, # how many prompts to be considered for assignment
            'empty_str': True
        },
        'demo': {
            'encoder': 'ada',
            'algo': 'kauto',
        }
    }
    return conf


def get_algo_conf(algo_conf):
    conf = globals()[algo_conf]()

    return conf
