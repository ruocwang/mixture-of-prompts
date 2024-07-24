'''
Taken from the Instruction Induction paper: https://arxiv.org/pdf/2205.10782.pdf
'''

import re
import string
from collections import Counter
import numpy as np
from src.misc import BBH_FMT_PROMPT 
from rouge_score import rouge_scorer
RougeScorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

# TODO: add some more metrics here for the new tasks.
TASK_TO_METRIC = {'common_concept': 'f1',
                  'informal_to_formal': 'f1',
                  'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es',
                  'synonyms': 'contains'
                  }
default_metric = 'em'


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction



def normalize_prediction_superni(pred, query, alphabets, task_format_type, lowercase=True, verbose=False):
    
    if verbose: print('-'*20 + "\n" + "pred_before : " + pred)

    pred = pred.strip()

    if task_format_type == 'Y':
        preds = pred.split('answer is')
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]
        
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]

    else:
        answer_flag = False
        pred = [pred]

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]

    if verbose: print("pred_after : " + pred)

    return pred


def normalize_prediction_superni_2(pred, query, alphabets, task_format_type, lowercase=True, verbose=True):
    
    if verbose: print('-'*20 + "\n" + "pred_before : " + pred)

    pred = pred.strip()

    preds = pred.split('answer is')
    answer_flag = True if len(preds) > 1 else False 
    pred = preds[-1]

    if task_format_type in ['ALPHABET_STR', 'N_LIST', 'F']:
        pred = [pred]

    elif task_format_type == 'N':
        # pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
        pred = [s for s in re.findall(r'-?\d+\.?\d*|[+*/-]', pred)]

    elif task_format_type in ['(M)_M']:
        done = False
        # (A)
        if done == False and len(re.findall(r'\([A-{}]\)'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'\([A-{}]\)'.format(alphabets[-1].upper()), pred)
            pred = [pred_.replace("(", "").replace(")", "").strip().upper() for pred_ in pred]
            done = True
        # A)
        if done == False and len(re.findall(r'[A-{}]\)'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'[A-{}]\)'.format(alphabets[-1].upper()), pred)
            pred = [pred_.replace(")", "").upper() for pred_ in pred]
            done = True
        # A
        if done == False and len(pred) == 1:
            pred = re.findall(r'[A-{}]'.format(alphabets[-1].upper()), pred)
            done = True
        if done == False:
            print('Fail to parse the answer')
            pred = [""]

    elif task_format_type in ['Option M:_Option M']:
        done = False
        # Option A
        if done == False and len(re.findall(r'Option [A-{}]'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'Option [A-{}]'.format(alphabets[-1].upper()), pred)
            done = True
        # Option (A)
        if done == False and len(re.findall(r'Option [\(A-{}\)]'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'Option [\(A-{}\)]'.format(alphabets[-1].upper()), pred)
            pred = [pred_.replace("(", "").replace(")", "") for pred_ in pred]
            done = True
        # Option A)
        if done == False and len(re.findall(r'Option [A-{}\)]'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'Option [A-{}\)]'.format(alphabets[-1].upper()), pred)
            pred = [pred_.replace(")", "") for pred_ in pred]
            done = True
        # (A)
        if done == False and len(re.findall(r'\([A-{}]\)'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'\([A-{}]\)'.format(alphabets[-1].upper()), pred)
            pred = ["Option " + pred_.replace("(", "").replace(")", "") for pred_ in pred]
            done = True
        # A)
        if done == False and len(re.findall(r'[A-{}]\)'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'[A-{}]\)'.format(alphabets[-1].upper()), pred)
            pred = ["Option " + pred_.replace(")", "").upper() for pred_ in pred]
            done = True
        # A
        if done == False and len(pred) == 1:
            pred = re.findall(r'[A-{}]'.format(alphabets[-1].upper()), pred)
            pred = ["Option " + pred_.upper() for pred_ in pred]
            done = True
        if done == False:
            print('Fail to parse the answer')
            pred = [""]

    elif task_format_type in ['m )_m']:
        done = False
        # a )
        if done == False and len(re.findall(r'[a-{}] \)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'[a-{}] \)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # a)
        if done == False and len(re.findall(r'[a-{}]\)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'[a-{}]\)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # a
        if done == False and len(pred) == 1:
            pred = re.findall(r'[a-{}]'.format(alphabets[-1].lower()), pred)
            done = True
        # a.
        if done == False and len(re.findall(r'[a-{}]\.'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'[a-{}]\.'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace(".", "").strip().lower() for pred_ in pred]
            done = True
        # (a)
        if done == False and len(re.findall(r'\([a-{}]\)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'\([a-{}]\)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("(", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # Option a )
        if done == False and len(re.findall(r'Option [a-{}] \)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}] \)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # Option a)
        if done == False and len(re.findall(r'Option [a-{}\)]'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}\)]'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # Option a
        if done == False and len(re.findall(r'Option [a-{}]'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}]'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").strip().lower() for pred_ in pred]
            done = True
        # Option a.
        if done == False and len(re.findall(r'Option [a-{}]\.'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}]\.'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace('.', '').strip().lower() for pred_ in pred]
            done = True
        # Option (a)
        if done == False and len(re.findall(r'Option [\(a-{}\)]'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [\(a-{}\)]'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace("(", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        if done == False:
            print('Fail to parse the answer')
            pred = [""]

    elif task_format_type in ['m._m']:
        done = False
        # a.
        if done == False and len(re.findall(r'[a-{}]\.'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'[a-{}]\.'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace(".", "").strip().lower() for pred_ in pred]
            done = True
        # a
        if done == False and len(pred) == 1:
            pred = re.findall(r'[a-{}]'.format(alphabets[-1].lower()), pred)
            done = True
        # a )
        if done == False and len(re.findall(r'[a-{}] \)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'[a-{}] \)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # a)
        if done == False and len(re.findall(r'[a-{}]\)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'[a-{}]\)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # (a)
        if done == False and len(re.findall(r'\([a-{}]\)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'\([a-{}]\)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("(", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # Option a.
        if done == False and len(re.findall(r'Option [a-{}]\.'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}]\.'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace('.', '').strip().lower() for pred_ in pred]
            done = True
        # Option a
        if done == False and len(re.findall(r'Option [a-{}]'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}]'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").strip().lower() for pred_ in pred]
            done = True
        # Option a )
        if done == False and len(re.findall(r'Option [a-{}] \)'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}] \)'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # Option a)
        if done == False and len(re.findall(r'Option [a-{}\)]'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [a-{}\)]'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        # Option (a)
        if done == False and len(re.findall(r'Option [\(a-{}\)]'.format(alphabets[-1].lower()), pred)) > 0:
            pred = re.findall(r'Option [\(a-{}\)]'.format(alphabets[-1].lower()), pred)
            pred = [pred_.replace("Option ", "").replace("(", "").replace(")", "").strip().lower() for pred_ in pred]
            done = True
        if done == False:
            print('Fail to parse the answer')
            pred = [""]

    elif task_format_type == 'Y':
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]

    else:
        raise ValueError("task type is not properly defined ...")


    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


def normalize_prediction_bbh(pred, query, alphabets, task_format_type, lowercase=True, verbose=True):
    if verbose: print('-'*20 + "\n" + "pred_before : " + pred)

    pred = pred.strip()

    preds = pred.split('answer is')
    answer_flag = True if len(preds) > 1 else False 
    pred = preds[-1]
    
    if task_format_type == 'N':
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    elif task_format_type == 'M':
        def get_options_str(query):
            options_str = query.split('\nOptions:\n')[-1].split('\nOutput:')[0].strip()
            return options_str
        
        def find_option_for_str(options_str, input_str):
            start_index = options_str.find(input_str)
            if start_index != -1:
                option_index = start_index -3
                return options_str[option_index]
            return None

        done = False
        # (A)
        if done == False and len(re.findall(r'\([A-{}]\)'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'\([A-{}]\)'.format(alphabets[-1].upper()), pred)
            done = True
        # A)
        if done == False and len(re.findall(r'[A-{}]\)'.format(alphabets[-1].upper()), pred)) > 0:
            pred = re.findall(r'[A-{}]\)'.format(alphabets[-1].upper()), pred)
            pred = ['(' + pred_.upper() for pred_ in pred]
            done = True
        if done == False and len(pred) == 1:
            pred = re.findall(r'[A-{}]'.format(alphabets[-1].upper()), pred)
            pred = ['(' + pred_ + ')' for pred_ in pred]
            done = True
        # when there's prediciton w/o option e.g., Ambiguous
        if done == False:
            options_str = get_options_str(query)
            option = find_option_for_str(options_str.lower(), pred.lower())
            if option is not None:
                pred = ['(' + option.upper() + ')']
                done = True

        if done == False:
            print('Fail to parse the answer')
            pred = [""]

    elif task_format_type == 'Y':
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]

    elif task_format_type == 'V':
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("valid", "invalid")]

    elif task_format_type == 'T':
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("true", "false")]

    elif task_format_type == 'S':
        # for "dick_languages task
        pred = pred.lower()
        pred = re.findall(r'[<>\{\}\[\]\(\)]', pred)
        pred = ' '.join(pred)
        pred = [pred]

    elif task_format_type == 'F':
        pred = pred.lower()
        pred = re.sub("\n|\.|\,","", pred)
        pred = pred.split(":")[-1].strip()
        pred = [pred]

    else:
        raise ValueError("task type is not properly defined ...")


    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    if verbose: print("pred_after : " + pred)

    return pred


def get_task_format(task_format_type, input_query=None, md_template=False):

    def alphabet_until(letter):
        letter = letter.lower()
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        return alphabet[:alphabet.index(letter) + 1]

    if not md_template:
        task_format = BBH_FMT_PROMPT[task_format_type]

    alphabets = None
    if task_format_type == 'M':
        first_alphabet = 'A'
        last_alphabet = input_query.split('(')[-1].split(')')[0]
        alphabets = alphabet_until(last_alphabet)
        task_format = task_format.replace('[FIRST]', first_alphabet).replace('[LAST]', last_alphabet)
    else:
        pass
    return task_format, alphabets


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    return prediction_normalized == ground_truth_normalized


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1


def get_bbh_score(prediction, ground_truth, verbose=False):
    def normalize(pred, lowercase=True):
        if type(pred) is str:
            pred = pred.strip()
            if lowercase:
                pred = pred.lower()
        else:
            import pdb; pdb.set_trace()
        return pred
    prediction_normalized = normalize(prediction, lowercase=True)
    ground_truth_normalized = normalize(ground_truth, lowercase=True)
    if verbose:
        print('='*20)
        print(f'prediction: {prediction}')
        print(f'ground_truth: {ground_truth}')
        print('-'*20)
        print(f'prediction: {prediction_normalized}')
        print(f'ground_truth: {ground_truth_normalized}')
    return prediction_normalized == ground_truth_normalized


def get_rouge_score(prediction, ground_truth, verbose=False):
    def normalize(pred, lowercase=False):
        pred = pred.strip()
        if lowercase:
            pred = pred.lower()
        return pred
    prediction_normalized = normalize(prediction)
    ground_truth_normalized = normalize(ground_truth)
    rouge_score = RougeScorer.score(
                    target=ground_truth_normalized, 
                    prediction=prediction_normalized)['rougeL'].fmeasure
    if verbose:
        print('-'*20)
        print(f'prediction: {prediction_normalized}')
        print(f'ground_truth: {ground_truth_normalized}')
        print(f'rouge_score: {rouge_score}')

    return rouge_score


def get_multi_answer_em(prediction, answers):
    for answer in answers:
        if get_em_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_f1(prediction, answers):
    f1_scores = []
    for answer in answers:
        f1_scores.append(get_f1_score(prediction, answer))
    return max(f1_scores)


def get_multi_answer_exact_set(prediction, answers):
    for answer in answers:
        if get_exact_set_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_contains(prediction, answers):
    for answer in answers:
        if get_contains_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_bbh(prediction, answers):
    for answer in answers:
        if get_bbh_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_rouge(prediction, answers):
    rouge_scores = []
    for answer in answers:
        rouge_scores.append(get_rouge_score(prediction, answer))
    return max(rouge_scores)