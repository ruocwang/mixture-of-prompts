import os
from src.exp_utils import json_load

def load_pregen_prompts(lib_name, task, seed=0, n_inits=1):
    if lib_name == 'empty':
        return ['' for _ in range(n_inits)]
    elif os.path.exists(lib_name):
        print(f'loading pregen from {lib_name}')
        pregen_prompts = json_load(lib_name)[task][str(seed)]
        return pregen_prompts
    else:
        return [globals()[lib_name][task][seed]]
