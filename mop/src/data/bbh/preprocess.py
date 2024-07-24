import os
import json
import random
from collections import defaultdict
from copy import deepcopy

random.seed(42)
induce_data_dir = os.path.join(os.path.dirname(__file__), 'raw/induce/')
eval_data_dir = os.path.join(os.path.dirname(__file__), 'raw/execute/')

## TODO under dev ##

# Get a list of tasks (by looking at the names of the files in the induced directory)
# tasks = [f.split('.')[0] for f in os.listdir(induce_data_path)]

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


def json_load(file_path):
    if not os.path.exists(file_path):
        print('loading empty json')
        return {}
    with open(file_path, "r") as outfile:
        obj = json.load(outfile)
    return obj


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    print(f'Read {len(result)} data from {path}')
    return result


def extract_common_sentence(sentence_list):
    # first sentence
    common_sentence = sentence_list[0]
    # search for common sentence by iterating over the rest of the sentences
    for sentence in sentence_list[1:]:
        common_sentence = ''.join(c1 for c1, c2 in zip(common_sentence, sentence) if c1 == c2)
    return common_sentence


def check_prefix_suffix(sentence, common):
    is_prefix, is_suffix = False, False
    # check if sentence is a prefix or suffix of common
    if sentence.startswith(common):
        is_prefix = True
    elif sentence.endswith(common):
        is_suffix = True
    return is_prefix, is_suffix


def preprocess_data(verbose=False, save=False):

    base_root = '/c1/sohyun/ape-ssl/ssl/src/data/bbh'
    base_directory = '/c1/sohyun/ape-ssl/ssl/src/data/bbh/source'
    big_bench_dir = '/c1/sohyun/BIG-bench/bigbench/benchmark_tasks'
    
    cnt = 0
    TASKS = []
    for root, dirs, files in os.walk(base_directory):
        for file_name in files:
            task_name = file_name.split('.')[0]
            file_path = os.path.join(root, file_name)
            
            if os.path.exists(file_path):
                cnt += 1
                TASKS.append(task_name)
                file_json = json_load(file_path)
                
                processed_data_induce = defaultdict(dict)
                processed_data_eval = defaultdict(dict)
                
                # metadata
                metadata = defaultdict(dict)
                
                # task.py
                if task_name in ['multistep_arithmetic_two', 'boolean_expressions', 'web_of_lies']:
                    if task_name == 'multistep_arithmetic_two':
                        name="multistep_arithmetic"
                        description="Solve multi-step arithmetic problems"
                        keywords=[
                            "mathematics",
                            "arithmetic",
                            "numerical response",
                            "zero-shot",
                            "multi-step",
                        ]
                        max_input_length_per_query=4
                        max_queries=5000
                    if task_name == 'boolean_expressions':
                        name="boolean_expressions"
                        description="Evaluate the result of a random Boolean expression"
                        keywords=[
                            "logical reasoning",
                            "multiple choice",
                            "computer code",
                            "algebra",
                            "non-language",
                            "multi-step",
                            "out of distribution",
                        ]
                        max_input_length_per_query=2048
                        max_queries=5000
                    if task_name == 'web_of_lies':
                        name="web_of_lies"
                        description="Evaluate a random boolean function expressed as a word problem"
                        keywords=[
                            "logical reasoning",
                            "multiple choice",
                            "context length",
                            "multi-step",
                            "out of distribution",
                        ]
                        max_input_length_per_query=2048
                        max_queries=5000
                    
                    key_list = ['name', 'canary', 'description', 'keywords', 'task_prefix', 'task_suffix', 'input_template']
                    for key in key_list:
                        if key == 'name':
                            metadata[key] = name
                        elif key == 'canary':
                            metadata[key] = file_json['canary']
                        elif key == 'description':
                            metadata[key] = description
                        elif key == 'keywords':
                            metadata[key] = keywords
                        elif key == 'task_prefix':
                            metadata[key] = ''
                        elif key == 'task_suffix':
                            metadata[key] = ''
                        elif key == 'input_template':
                            metadata[key] = ''

                # task.json
                else:
                    big_bench_file_path = os.path.join(big_bench_dir, task_name, 'task.json')
                    
                    if task_name == 'logical_deduction_three_objects':
                        big_bench_file_path = os.path.join(big_bench_dir, 'logical_deduction', 'three_objects', 'task.json')
                    if task_name == 'logical_deduction_five_objects':
                        big_bench_file_path = os.path.join(big_bench_dir, 'logical_deduction', 'five_objects', 'task.json')
                    if task_name == 'logical_deduction_seven_objects':
                        big_bench_file_path = os.path.join(big_bench_dir, 'logical_deduction', 'seven_objects', 'task.json')
                    if task_name == 'tracking_shuffled_objects_three_objects':
                        big_bench_file_path = os.path.join(big_bench_dir, 'tracking_shuffled_objects', 'three_objects', 'task.json')
                    if task_name == 'tracking_shuffled_objects_five_objects':
                        big_bench_file_path = os.path.join(big_bench_dir, 'tracking_shuffled_objects', 'five_objects', 'task.json')
                    if task_name == 'tracking_shuffled_objects_seven_objects':
                        big_bench_file_path = os.path.join(big_bench_dir, 'tracking_shuffled_objects', 'seven_objects', 'task.json')
                    if task_name == 'formal_fallacies':
                        big_bench_file_path = os.path.join(big_bench_dir, 'formal_fallacies_syllogisms_negation', 'task.json')
                    
                    if os.path.exists(big_bench_file_path):
                        big_bench_file_json = json_load(big_bench_file_path)
                    else:
                        print(task_name)
                        print(f'>>> {big_bench_file_path} is not in BIG-bench')
                        import pdb; pdb.set_trace()
                    key_list = ['name', 'canary', 'description', 'keywords', 'task_prefix', 'task_suffix', 'input_template']
                    for key in key_list:
                        if key in big_bench_file_json:
                            metadata[key] = big_bench_file_json[key]
                        else:
                            metadata[key] = ''

                metadata_induce = deepcopy(metadata)
                metadata_eval = deepcopy(metadata)

                # examples
                examples = file_json['examples']
                # indices for processed_data_induce
                indices = random.sample(range(len(examples)), int(len(examples)//2))
                metadata_induce['num_examples'] = len(indices)
                metadata_eval['num_examples'] = len(examples) - len(indices)
                
                examples_induce = defaultdict(dict)
                examples_eval = defaultdict(dict)
                input_list_induce = []
                input_list_eval = []

                cnt_induce, cnt_eval = 0, 0
                for i, ex in enumerate(examples):
                    if i in indices:
                        examples_induce['examples'][str(cnt_induce + 1)] = {
                            'input': ex['input'],
                            'output': ex['target']
                        }
                        input_list_induce.append(ex['input'])
                        cnt_induce += 1
                    else:
                        examples_eval['examples'][str(cnt_eval + 1)] = {
                            'input': ex['input'],
                            'output': ex['target']
                        }
                        input_list_eval.append(ex['input'])
                        cnt_eval += 1
                
                if task_name in ['']:
                
                    # extract common sentence from input list
                    common_sentence_induce = extract_common_sentence(input_list_induce)
                    common_sentence_eval = extract_common_sentence(input_list_eval)
                    assert common_sentence_induce == common_sentence_eval, f"common_sentence_induce: {common_sentence_induce}, common_sentence_eval: {common_sentence_eval}"
                    # check the common sentence
                    if common_sentence_induce in ["Today, ", ""]:
                        common_sentence_induce = ""
                        common_sentence_eval = ""
                    print('='*40)
                    print('common sentence')
                    print(common_sentence_induce)
                
                    is_prefix, is_suffix = check_prefix_suffix(input_list_induce[0], common_sentence_induce)
                    if is_prefix:
                        if metadata_induce['task_prefix'] != '' and metadata_induce['task_prefix'] != common_sentence_induce:
                            import pdb; pdb.set_trace()
                        metadata_induce['task_prefix'] = common_sentence_induce
                    if is_suffix:
                        metadata_induce['task_suffix'] = common_sentence_induce
                    is_prefix, is_suffix = check_prefix_suffix(input_list_eval[0], common_sentence_eval)
                    if is_prefix:
                        metadata_eval['task_prefix'] = common_sentence_eval
                    if is_suffix:
                        metadata_eval['task_suffix'] = common_sentence_eval
                    import pdb; pdb.set_trace()
                    
                
                # remove the task prefix or task suffix from the examples
                # if metadata_induce['task_prefix'] != '':
                #     for i, ex in enumerate(examples_induce):
                #         examples_induce[i]['input'] = ex['input'].replace(metadata_induce['task_prefix'], '')
                
                
                # full processed data
                processed_data_induce['metadata'] = metadata_induce
                processed_data_eval['metadata'] = metadata_eval
                processed_data_induce['examples'] = examples_induce
                processed_data_eval['examples'] = examples_eval

                if save:
                    # save path
                    induce_data_path = induce_data_dir + task_name + '.json'
                    json_save(processed_data_induce, induce_data_path)
                    if verbose: print(f'>>> Finish saving induce data to {induce_data_path}')

                    # save path
                    eval_data_path = eval_data_dir + task_name + '.json'
                    json_save(processed_data_eval, eval_data_path)
                    if verbose: print(f'>>> Finish saving eva; data to {eval_data_path}')

    if verbose: print(f'>>> Finish processing {cnt} tasks')

    if save:
        txt_file_path = base_root + 'TASKS_BBH.txt' 
        with open(txt_file_path, 'w') as file:
            for task in TASKS:
                file.write(f"{task}\n")
        if verbose: print(f'>>> Finish saving {txt_file_path}')



if __name__ == '__main__':
    preprocess_data()