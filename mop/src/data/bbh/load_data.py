import os
import json
import random
from src.exp_utils import directories

induce_data_path = os.path.join(directories.benchmarks_dir, 'bbh/raw/induce/')
eval_data_path = os.path.join(directories.benchmarks_dir, 'bbh/raw/execute/')
induce_data_path_ori = os.path.join(directories.benchmarks_dir, 'bbh/raw_original/induce/')
eval_data_path_ori = os.path.join(directories.benchmarks_dir, 'bbh/raw_original/execute/')

# Get a list of tasks (by looking at the names of the files in the induced directory)
tasks = [f.split('.')[0] for f in os.listdir(induce_data_path)]


def load_data_bbh(type, task, opt_desc=True):
    if opt_desc:
        base_path = induce_data_path_ori if type == 'induce' else eval_data_path_ori
    else:
        base_path = induce_data_path if type == 'induce' else eval_data_path
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []

    for i in range(num_examples):
        data = examples[str(i + 1)]
        input_, output_ = data['input'], [data['output']]
        inputs.append(input_)
        outputs.append(output_)

    return inputs, outputs
