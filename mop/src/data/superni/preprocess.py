import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

induce_data_dir = os.path.join(os.path.dirname(__file__), 'raw/induce/')
eval_data_dir = os.path.join(os.path.dirname(__file__), 'raw/execute/')

SEED = 0

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


def json_save(obj, file_path):
    json_object = json.dumps(obj, indent=4)
    json_object = unindent_list(json_object)
    with open(file_path, "w") as outfile:
        outfile.write(json_object)


def json_load(file_path, verbose=False):
    if not os.path.exists(file_path):
        if verbose: print('loading empty json')
        return {}
    with open(file_path, "r") as outfile:
        obj = json.load(outfile)
    return obj


def read_jsonl_as_list(path: str, verbose=False):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    if verbose: print(f'Read {len(result)} data from {path}')
    return result


def preprocess_data():

    base_directory = '/c1/sohyun/mop-icml/mop/src/data/superni/source'
    
    ########## DEBUG (START)##########
    # # training: 574 / total: 665
    # total = 0
    # n_tr = 0
    # for root, dirs, files in os.walk(base_directory):
    #     for dir_name in dirs:
    #         if dir_name.startswith('task'):
    #             total += 1
    #             dir_path = os.path.join(root, dir_name)
    #             train_examples_json_file_path = os.path.join(dir_path, 'train_examples.jsonl')
    #             if os.path.exists(train_examples_json_file_path): n_tr += 1
    ########## DEBUG (END)##########
    
    cnt = 0
    TASKS = []
    for root, dirs, files in os.walk(base_directory):
        for dir_name in tqdm(dirs):
            if dir_name.startswith('task'):
                
                dir_path = os.path.join(root, dir_name)
                all_examples_json_file_path = os.path.join(dir_path, 'all_examples.jsonl')
                task_metadata_json_file_path = os.path.join(dir_path, 'task_metadata.json')
                demo_template_json_file_path = os.path.join(dir_path, 'demo_template.json')
                
                if os.path.exists(all_examples_json_file_path):
                    cnt += 1
                    TASKS.append(dir_name)
                    
                    all_examples = read_jsonl_as_list(all_examples_json_file_path)

                    # create split
                    random.seed(SEED)
                    indices = random.sample(range(len(all_examples)), len(all_examples)//2)
                    induce_examples = [all_examples[i] for i in indices]
                    eval_examples = [all_examples[i] for i in range(len(all_examples)) if i not in indices]
                    
                    ##### induce_data
                    processed_data_induce = defaultdict(dict)
                    # examples
                    examples_induce = defaultdict(dict)
                    for i, induce_example in enumerate(induce_examples):
                        examples_induce[str(i + 1)] = {
                            "input": induce_example["input"],
                            "output": induce_example["target"],
                        }
                    # metadata
                    metadata_induce = json_load(task_metadata_json_file_path)
                    metadata_induce["num_examples"] = len(induce_examples)
                    processed_data_induce["metadata"] = metadata_induce
                    processed_data_induce["demo_template"] = json_load(demo_template_json_file_path)
                    processed_data_induce["examples"] = examples_induce
                    
                    ##### eval_data
                    processed_data_eval = defaultdict(dict)
                    # examples
                    examples_eval = defaultdict(dict)
                    for i, eval_example in enumerate(eval_examples):
                        examples_eval[str(i + 1)] = {
                            "input": eval_example["input"],
                            "output": eval_example["target"],
                        }
                    # metadata
                    metadata_eval = json_load(task_metadata_json_file_path)
                    metadata_eval["num_examples"] = len(eval_examples)
                    processed_data_eval["metadata"] = metadata_eval
                    processed_data_eval["demo_template"] = json_load(demo_template_json_file_path)
                    processed_data_eval["examples"] = examples_eval
                    
                    # save induce data
                    induce_data_path = induce_data_dir + dir_name + '.json'
                    json_save(processed_data_induce, induce_data_path)

                    # save eval data
                    eval_data_path = eval_data_dir + dir_name + '.json'
                    json_save(processed_data_eval, eval_data_path)
                else:
                    import pdb; pdb.set_trace()

    print(f'>>> Finish processing {cnt} tasks')

    txt_file_path = '/c1/sohyun/mop-icml/mop/src/data/superni/TASKS_SuperNI.txt' 
    with open(txt_file_path, 'w') as file:
        for task in TASKS:
            file.write(f"{task}\n")
    print(f'>>> Finish saving {txt_file_path}')


if __name__ == '__main__':
    preprocess_data()