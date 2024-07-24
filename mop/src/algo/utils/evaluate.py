import numpy as np
import logging

from abc import ABC, abstractmethod
from algo.utils import data, llm
from src.evaluation.instruction_induction import utility
from src.misc import BBH_TASKS, SUPERNI_TASKS
from collections import defaultdict


def get_eval_method(eval_method):
    """
    Returns the evaluation method object.
    Parameters:
        eval_method: The evaluation method to use. ('likelihood')
    Returns:
        An evaluation method object.
    """
    if callable(eval_method):
        return eval_method
    if eval_method == 'likelihood':
        from algo.automatic_prompt_engineer.evaluation import likelihood
        return likelihood.likelihood_evaluator
    elif eval_method == 'bandits':
        from algo.automatic_prompt_engineer.evaluation import bandits
        return bandits.bandits_evaluator
    else:
        raise ValueError('Invalid evaluation method.')


def evaluate_prompts(prompters, data_man, config, seed=None):
    return exec_accuracy_evaluator(prompters, data_man, config, seed=seed)


def demo_function(temp_man, config):
    """
    Returns a function that can be manually test the LLM with a chosen prompt.
    Parameters:
        config: The configuration dictionary.
    Returns:
        A function that takes a prompt and returns a demo.
    """
    model = llm.model_from_config(config['model'])

    def fn(prompt, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        queries = []
        for input_ in inputs:
            query = temp_man.eval.fill(prompt=prompt, input=input_)
            queries.append(query)
        outputs = model.generate_text(
            queries, n=1)
        return [out.strip().split('\n')[0] for out in outputs]

    return fn


#######################################################
#################### eval function ####################
#######################################################
def exec_accuracy_evaluator(prompters, data_man, config, seed=None):
    """
        prompters: (a list of) prompter(s) that support forward()
        deter (poor name): use all evaluation data
        generator: use the provided python random generator for sampling a fixed set of subsamples
    """
    if not isinstance(prompters, list):
        prompters = [prompters]

    ## Instantiate the LLM
    model = llm.model_from_config(config['model'])

    ## prepare eval data & predict
    input_pred_ans_list = []
    for prompter in prompters:
        inputs = []
        answers = []
        demo_datas = []

        subsampled_data = data.subsample_data(data_man.get_split(config['split']), config['num_samples'])

        for d in zip(*subsampled_data):
            input_, output_ = d
            demo_data = data.subsample_data(data_man.few_shot_data, config['num_few_shot'], seed=seed)
            demo_datas.append(demo_data)
            inputs.append(input_)
            answers.append(output_)
        input_pred_ans_list.append([*prompter.forward(model, demo_datas, inputs, answers, config['add_icls'])])


    ## load metrics
    task = config['task']
    if task in BBH_TASKS:
        metric = 'bbh'
    elif task in SUPERNI_TASKS:
        metric = 'rouge'
    else:
        metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)

    if metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em
    elif metric == 'cot':
        score_fn = utility.get_multi_answer_cot
    elif metric == 'bbh':
        score_fn = utility.get_multi_answer_bbh
    elif metric == 'rouge':
        score_fn = utility.get_multi_answer_rouge

    ## evaluate
    scores = []
    mispredicts = defaultdict(list)
    for pid, input_pred_ans_pp in enumerate(input_pred_ans_list):
        for input_, pred_, ans_ in zip(*input_pred_ans_pp):
            score = score_fn(pred_, ans_)
            if score == 0:
                pred_ = pred_.replace('\n', '')
                mispredicts[pid].append([input_, pred_, ans_])
            scores.append(score)

    # Reshape the scores so that it is num_prompts x num_samples
    scores = np.array(scores).reshape(len(prompters), -1)
    # logging.info(f'scores: {scores}')

    res = ExecAccuracyEvaluationResult(prompters, scores)
    return res, scores, mispredicts


class EvaluationResult(ABC):

    @abstractmethod
    def sorted(self, method='default'):
        """Get the results in the form of a sorted prompt and score list.
        Has a method argument to support sorting by various metrics."""
        pass

    @abstractmethod
    def in_place(self, method='default'):
        """Get the results in the form of a list of prompts and scores without sorting."""
        pass


class ExecAccuracyEvaluationResult(EvaluationResult):

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""
        if method == 'mean':
            return [np.mean(s) for s in self.scores]
        elif method == 'median':
            return [np.median(s) for s in self.scores]
        elif method == 'std':
            return [np.std(s) for s in self.scores]
        elif method == 'max':
            return [np.max(s) for s in self.scores]
        elif method == 'min':
            return [np.min(s) for s in self.scores]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.scores]
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def sorted(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        return self.prompts, scores
