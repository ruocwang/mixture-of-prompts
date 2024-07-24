"""
    prompt engineering manager
"""
import numpy as np
import random
from tqdm import tqdm
from src.evaluation.instruction_induction.utility import normalize_prediction, normalize_prediction_bbh, normalize_prediction_superni, get_task_format
from src.exp_utils import json_load, json_save
from src.misc import BBH_TASKS_FMT


def add_demos_to_prompt(prompt, demos, temp_man, max_length=None):
    prompt += '\n\n' + temp_man.demos.fill(demos)
    if max_length is not None:
        if len(prompt) > max_length:
            prompt = prompt[:max_length]
            prompt = prompt[:prompt.rfind('Input:')]
    return prompt


class Prompter():
    """
        base class, the only place that takes a list of string prompts as input
    """
    def __init__(self, prompts, config, temp_man):
        """
            Input:
                prompts: (a list of) string prompts
        """
        self.config = config
        self.args = self.config['evaluation']['args']
        self.temp_man = temp_man

        self.prompts = self.init_prompts(prompts)

        self._set_task_info()


    def __hash__(self):
        """ Use a hash of the id attribute to define uniqueness """
        return hash(str(self))


    def __eq__(self, other):
        """ Compare objects based on the list of prompts it holds """
        return str(self)


    def __str__(self) -> str:
        """ used in print(Prompter) or str(Prompter) """
        return ' || '.join(self.prompts)


    def init_prompts(self, prompts):
        """ a prompter object can be initialized from a list of prompts or prompters """
        if not isinstance(prompts, list):
            prompts = [prompts]
        if isinstance(prompts[0], str):
            return prompts
        else: ## passed in prompter objects
            return [str(prompter) for prompter in prompts]

    def _set_task_info(self):
        try:
            args = self.config['args']
        except:
            for k in list(self.config.keys()):
                if 'args' in self.config[k]: 
                    args = self.config[k]['args']
                    break

        self.benchmark = args.benchmark
        self.task = args.task
        self.task_format_type = None
        if self.benchmark == 'bbh':
            self.task_format_type = BBH_TASKS_FMT[self.task]

    def set_prompts(self, new_prompts):
        if not isinstance(new_prompts, list):
            new_prompts = [new_prompts]
        assert len(new_prompts) == len(self.prompts), 'number of prompts must align'
        self.prompts = new_prompts


    def add_demos(self, demos):
        self.prompts[0] += '\n\n' + f'{self.temp_man.demos.fill(demos)}'


    def forward(self, model, demo_datas, query_inputs, query_answers, add_icls=False, is_test=False):
        """
            Input:
                demo_datas: a list of few-shot demos (not used in APE)
                query_inputs: a set of inputs
                query_answers: ground-truth, used in pens-oracle
        """
        assert len(demo_datas) == len(query_inputs)
        assert len(self.prompts) == 1
        prompt = self.prompts[0]

        ## generate queries
        queries = []
        alphabets_list = [] # newly added
        for demo_data, query_input in zip(demo_datas, query_inputs):
            query, alphabets = self.get_query(prompt, query_input, demo_data, add_icls)
            queries.append(query)
            alphabets_list.append(alphabets) # newly added

        ## inference
        model_outputs = model.generate_text(queries, 1)
        model_outputs = self.normalize_prediction(model_outputs, queries, alphabets_list)

        ## ensemble
        model_outputs = np.array(model_outputs).tolist()

        return query_inputs, model_outputs, query_answers

    def get_query(self, prompt, input_, demo_data, add_icls=False):
        demos = self.temp_man.demos.fill(demo_data)
        if add_icls != 0:
            prompt = add_demos_to_prompt(prompt, demo_data, self.temp_man)

        alphabets = None
        if self.benchmark == 'bbh':
            _, alphabets = get_task_format(self.task_format_type, input_)

        query = self.temp_man.eval.fill(prompt=prompt, input=input_, output='', full_demo=demos)
        
        return query, alphabets


    def normalize_prediction(self, preds, queries, alphabets_list):
        """ some hardcoded postprocessing from IZ codebase. """
        if self.args.benchmark == 'bbh':
            return [normalize_prediction_bbh(pred, query, alphabets, self.task_format_type, lowercase=True)
                    for pred, query, alphabets in zip(preds, queries, alphabets_list)]
        elif self.args.benchmark == 'superni':
            return [normalize_prediction_superni(pred, query, alphabets, self.task_format_type, lowercase=True)
                    for pred, query, alphabets in zip(preds, queries, alphabets_list)]

        else:
            return [normalize_prediction(pred, lowercase=True) for pred in preds]

    def add_icls(self, icl_splits):
        assert len(icl_splits) == len(self.prompts)

        for pid in range(len(self.prompts)):
            self.prompts[pid] = add_demos_to_prompt(self.prompts[pid], icl_splits[pid],
                                                    self.temp_man, max_length=self.max_length)


class PrompterMoP(Prompter):
    """
        Prompter for MoP
    """

    def __init__(self, prompts, demos_pack, config, temp_man, eval_id=None):
        """
            Input:
                prompts: (a list of) string prompts
                eval_id: specifically evaluate one expert, other expert will output oracle when routed
        """
        super().__init__(prompts, config, temp_man)

        self.config = config
        self.demos_splits = demos_pack[0]
        self.assign_partition_fn = demos_pack[1]
        self.get_query_fn = demos_pack[2] if len(demos_pack) > 2 else None
        self.eval_num_samples = config['evaluation']['num_samples'] if len(demos_pack) > 2 else None
        self.eval_threshold = config['evaluation']['threshold'] if len(demos_pack) > 2 else None
        self.make_prompter = lambda prompts: Prompter(prompts, config, temp_man)
        
        self.set_eval_id(eval_id)


    def save(self, path):
        """ save prompts """
        prompts_dict = json_load(path)
        for eid in range(len(self.prompts)):
            prompt = f'{self.prompts[eid]}' + '\n\n' + f'{self.temp_man.demos.fill(self.demos_splits[eid])}'
            final_prompt = self.temp_man.eval.fill(prompt=prompt, input='', output='', full_demo='')
            prompts_dict[eid] = final_prompt
        json_save(prompts_dict, path)


    def __str__(self) -> str:
        """ used in print(Prompter) or str(Prompter) """
        seppa = '\n' + '-'*30 + '\n'
        mes = 'MixtureOfPrompts\n'
        mes += '='*40 + '\n'
        mes += seppa.join([f'[EXPERT]: {self.prompts[eid]}' for eid in range(len(self.prompts))]) + '\n'
        mes += '='*40 + '\n'
        full = [f'{self.prompts[eid]}' + '\n\n' + f'{self.temp_man.demos.fill(self.demos_splits[eid])}' for eid in range(len(self.prompts))]
        mes += seppa.join([f'[Full Prompt]: {full[eid]}' for eid in range(len(self.prompts))]) + '\n'
        mes += '='*40
        return mes

    def __len__(self) -> int:
        return len(self.prompts)

    def set_eval_id(self, eval_id):
        if eval_id is not None:
            self.eval_id = eval_id if isinstance(eval_id, list) else [eval_id]
        else:
            self.eval_id = list(range(len(self.prompts)))

    def routing(self, new_data=None, expert_id=None, return_idx=False):
        """ route the new test point to an expert """
        if expert_id is None:
            expert_id = self.assign_partition_fn(new_data)

        prompt = self.prompts[expert_id] + '\n\n' + self.temp_man.demos.fill(self.demos_splits[expert_id])
        prompter = self.make_prompter(prompt)

        if return_idx:
            return prompter, expert_id
        else:
            return prompter
    
    def add_query(self, queries, alphabets_list, query_inputs_, query_answers_, is_evals, demo_datas, query_inputs, query_answers, prompter):
        """ if not enough eval data for some expert, add more eval data """
        assert len(self.eval_id) == 1
        expert_id  = self.eval_id[0]
        query_indices = list(self.get_query_fn(expert_id, query_inputs))
        
        for idx in query_indices:
            demo_data = demo_datas[idx]
            query_input = query_inputs[idx]
            query_answer = query_answers[idx]
            if query_input not in query_inputs_:
                query, alphabets = self.get_query(prompter.prompts[0], query_input, demo_data)
                queries.append(query)
                alphabets_list.append(alphabets)
                query_inputs_.append(query_input)
                query_answers_.append(query_answer)
                is_evals.append(True)
            if np.sum(is_evals) >= self.eval_num_samples:
                break
        return queries, alphabets_list, query_inputs_, query_answers_, is_evals

    def forward(self, model, demo_datas, query_inputs, query_answers, add_icls=False, fixed_agent_id=-1):
        """
            forward fn uses memoization to avoid duplicated query for embedding models
            Input:
                demo_datas: a list of few-shot demos (not used in APE)
                query_inputs: a set of inputs
                query_answers: ground-truth, used in evaluation
            Output:
                query_inputs_: a subset of `query_inputs` belonging to expert_id
                model_outputs: model predictions for `query_inputs_`
                query_answers_: ground-truth for `query_inputs_`
        """
        assert len(demo_datas) == len(query_inputs) == len(query_answers)

        ## generate queries
        queries = []
        alphabets_list = []
        query_inputs_ = []
        query_answers_ = []
        is_evals = []
        for demo_data, query_input, query_answer in tqdm(zip(demo_datas, query_inputs, query_answers)):
            prompter, expert_id = self.routing(query_input, return_idx=True)
            if expert_id in self.eval_id:
                query, alphabets = self.get_query(prompter.prompts[0], query_input, demo_data, add_icls)
                queries.append(query)
                alphabets_list.append(alphabets)
                query_inputs_.append(query_input)
                query_answers_.append(query_answer)
            is_evals.append(expert_id in self.eval_id)

        # add more eval samples when there's not enough evals
        if self.eval_threshold and np.sum(is_evals) < self.eval_num_samples and len(self.eval_id) == 1:
            queries, alphabets_list, query_inputs_, query_answers_, is_evals = self.add_query(queries, alphabets_list, query_inputs_, query_answers_, is_evals, demo_datas, query_inputs, query_answers, prompter)

        ## inference
        model_outputs = model.generate_text(queries, 1)
        model_outputs = self.normalize_prediction(model_outputs, queries, alphabets_list)

        return query_inputs_, model_outputs, query_answers_
