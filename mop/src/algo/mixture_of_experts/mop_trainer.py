import os
import sys
import random
import torch
import numpy as np
import logging

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from algo.utils.models import GPT2, OpenAIModels, SentenceT5
from copy import deepcopy
from src.pe_manager import PrompterMoP, Prompter
from .utils import chooseBestKforKMeansParallel


import numpy as np
from src.algo.utils import generate, evaluate


class MoPTrainer():

    def __init__(self, n_experts, conf, temp_man):
        self.max_n_experts = n_experts
        self.n_experts = n_experts
        self.conf = conf
        self.temp_man = temp_man

        self.make_prompter_mop = lambda prompts, demos_pack: PrompterMoP(prompts, demos_pack, conf, temp_man)
        self.partition_indices_list = None
        self.cluster_assignments = None


    def train_experts(self, data_man, eval_fn, pgen_fn):
        """ Main train function: build expert by assigning demos and prompts """
        """
            input:
                data_man: data manager class
                eval_fn: evaluate a prompter on eval_data
                pgen_fn: pgen_fn(pgen_data) generates (and evals) prompters in decendent order
            return:
                assign_partition_fn: assign a new data point to an expert
        """

        #### generate demos
        demos_pack, trimmed_demos = self.assign_demos(data_man.few_shot_data)
        
        #### Region-based joint search
        ## generate instructions
        prompters_pack = self.generate_inst(data_man, demos_pack, trimmed_demos, pgen_fn)
        ## assign instructions
        prompters = self.assign_inst(prompters_pack, demos_pack, eval_fn)
        
        return self.make_prompter_mop(prompters, demos_pack)


    def _gen_prompter(self, partitions):
        prompters = []
        for partition in partitions:
            prompters += generate.generate_prompts(partition, self.temp_man, self.conf)
        return prompters


    def assign_demos(self, demos):
        logging.info('='*20 + ' [Assigning demos] ' + '='*20 + '\n')
        
        demo_conf = self.conf['method']['demo']
        if demo_conf['algo'] == 'random':
            demos_pack, trimmed_demos = self._random_partition(demos, self.max_n_experts), None

        elif demo_conf['algo'] == 'fix':
            demos_pack, trimmed_demos = self._fixed_partition(demos, self.max_n_experts), None

        elif demo_conf['algo'] in ['kauto', 'kmean', 'kmean-constrained']:
            demos_pack, trimmed_demos = self._cluster_partition(demos, self.max_n_experts, demo_conf)

        else:
            raise ValueError(demo_conf['algo'])
        
        # identify partition size
        partitions_len = []
        partitions = demos_pack[0]
        for partition in partitions:
            partitions_len.append(len(partition[0]))
        logging.info(f'partition size: {partitions_len}')

        return demos_pack, trimmed_demos
    

    def generate_inst(self, data_man, demos_pack, trimmed_demos, pgen_fn):
        """
        Generates prompters for the Mixture of Prompts (MoP) training algorithm.

        Args:
            trimmed_demos: demos that gets trimmed by kauto.
            pgen_fn (function): The prompt generator function.
                This function takes two inputs: prompters and conf, and returns a list of prompters (and their scores).

        Returns:
            Prompters_pack: A list of [prompters] for each expert, similar to demos_pack.
                This was originally designed for assigning different candidates to different experts. But it is not required since I found that sharing candidates across experts works better.

        """
        logging.info('='*20 + ' [Generating prompts] ' + '='*20 + '\n')
        inst_conf = self.conf['method']['inst']
        if inst_conf['pgen'] == 'unconditional':  ## MoP v1 (original version)
            prompters = pgen_fn(data_man.prompt_gen_data, self.conf)
            prompters = prompters[:inst_conf['n_inits']]
            prompters_pack = [deepcopy(prompters) for _ in range(len(demos_pack[0]))]

        elif inst_conf['pgen'] == 'compensate':  ## MoP v2
            prompters, scores = [], []
            partitions = demos_pack[0]
            
            ## adjust num_subsamples in conf so that the total budget is no higher than ape20.
            adjusted_conf = deepcopy(self.conf)
            adjusted_conf['generation']['num_subsamples'] *= self.max_n_experts // self.n_experts

            for i in range(len(partitions)):
                ## use demos from all other partitions to generate prompts
                complements = partitions[:i] + partitions[i + 1:] if i + 1 < len(partitions[0]) else partitions[:i]
                merged_complements = [[], []]
                for complement in complements:
                    merged_complements[0] += complement[0]
                    merged_complements[1] += complement[1]
                if trimmed_demos is not None:
                    merged_complements[0] += trimmed_demos[0]
                    merged_complements[1] += trimmed_demos[1]
                ret = pgen_fn(tuple(merged_complements), adjusted_conf, return_score=True)
                prompters += ret[0]
                scores += ret[1]
                logging.info(f'>>> generated {len(ret[0])} prompts from pid {i:>2d} with score={ret[1]}')

            ## merge all candidate prompts together, sort and pick top n_inits
            res = evaluate.ExecAccuracyEvaluationResult(prompters, scores)
            prompters = res.sorted()[0][:inst_conf['n_inits']]
            prompters_pack = [deepcopy(prompters) for _ in range(self.n_experts)]

        else:
            raise NotImplementedError

        ## postprocessing prompters_pack
        if inst_conf['algo'] == 'top1':
            prompters_pack = [[prompters[0]] for prompters in prompters_pack]
        elif inst_conf['empty_str']:  # add empty prompter to the candidate space
            for prompters in prompters_pack:
                prompters.append(Prompter([''], self.conf, self.temp_man))

        return prompters_pack


    def assign_inst(self, prompters_pack, demos_pack, eval_fn):
        logging.info('='*20 + ' [Assigning instructions] ' + '='*20 + '\n')
        assert len(prompters_pack) == len(demos_pack[0])
        inst_conf = self.conf['method']['inst']
        if inst_conf['algo'] == 'is': # independent search
            prompters = self._independent_search(prompters_pack, demos_pack, eval_fn)

        elif inst_conf['algo'] == 'js':
            prompters = self._joint_search(prompters_pack, demos_pack, eval_fn, region=False)

        elif inst_conf['algo'] == 'rbjs':
            prompters = self._joint_search(prompters_pack, demos_pack, eval_fn, region=True)

        else:
            raise ValueError(inst_conf['algo'])

        return [prompt for prompt in prompters]


    #######################################################
    ################ instruction assignment ###############
    #######################################################
    def _independent_search(self, prompters_pack, demos_pack, eval_fn):
        best_prompters = [deepcopy(prompters[0]) for prompters in prompters_pack]
        return best_prompters


    def _joint_search(self, prompters_pack, demos_pack, eval_fn, replacement=True, region=True):
        ## build the initial mop
        best_prompters = [deepcopy(prompters[0]) for prompters in prompters_pack] # init mop with 1st prompter

        for eid in range(self.n_experts):
            ## find the prompt for expert eid
            best_prompter, best_score, best_pid = None, 0.0, 0
            for pid, prompter in enumerate(prompters_pack[eid]):
                best_prompters[eid] = deepcopy(prompter)
                if region:
                    prompter_mop = self.make_prompter_mop(best_prompters, demos_pack)
                    prompter_mop.set_eval_id(eid)  ## only evaluate this expert, save budget
                    score = eval_fn(prompter_mop)
                else:
                    best_prompters[eid].add_demos(demos_pack[0][eid])
                    score = eval_fn(best_prompters[eid])
                # logging.info(f'>>> evaluating [eid: {eid:>2d} | pid: {pid:>2d}] with score={score}\n')
                if score > best_score:
                    best_prompter, best_score = deepcopy(prompter), score
                    best_pid = pid

            ## assign best prompter to expert i
            logging.info(f'best prompt for expert={eid:>2d}: pid={best_pid:>2d} score={best_score}')
            best_prompters[eid] = best_prompter

            if not replacement:
                prompters = [prompter for prompter in prompters if prompter.prompts[0] != best_prompter.prompts[0]]

        logging.info('='*20 + ' Completed ' + '='*20)

        return best_prompters


    #######################################################
    ################### demo assignment ###################
    #######################################################
    def _cluster_partition(self, demos, N, demo_conf, prefix=''):
        """
            demos: ([inputs], [outputs])
        """
        ## text embedding model
        if demo_conf['encoder'] == 'ada':
            text_encoder = OpenAIModels(model='text-embedding-ada-002',
                                        benchmark=self.conf['evaluation']['args'].benchmark)
        elif demo_conf['encoder'] == 'gpt2-large':
            text_encoder = GPT2('gpt2-large')
        elif demo_conf['encoder'] == 't5':
            text_encoder = SentenceT5('t5-base')
        elif demo_conf['encoder'] == 't5-l':
            text_encoder = SentenceT5('t5-large')
        elif demo_conf['encoder'] == 't5-xl':
            text_encoder = SentenceT5('t5-xl')
        else:
            raise ValueError(demo_conf['encoder'])

        #### clustering
        data_matrix = self._demos_to_embeds(demos, text_encoder)

        args = self.conf['generation']['args']
        if demo_conf['algo'] == 'kauto':
            ## choose the best k
            best_k, _ = chooseBestKforKMeansParallel(data_matrix,
                            k_range=range(2, max(3, min(N, data_matrix.shape[0]) + 1)),
                            seed=args.seed)
            self.n_experts = best_k # update n_experts
            print(f'==> best_k = {best_k}...')

            kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=args.seed)
            kmeans.fit(data_matrix)
            predict_fn = lambda data: kmeans.predict(data)
            cluster_assignments = kmeans.labels_

        #### enforce the maximum number of demos in each partition
        partitions = []
        partition_indices_list = []
        partition_size = len(demos[0]) // N

        # Special case for length limit
        args = self.conf['generation']['args']
        if args.benchmark == 'bbh' and args.task == 'causal_judgement' and partition_size > 6:
            partition_size = 6

        trimmed_demos = [[], []]
        for i in range(self.n_experts):
            partition_indices = np.where(cluster_assignments == i)[0]
            
            if len(partition_indices) > partition_size:
                saved_partition_indices = np.array(random.sample(list(partition_indices), partition_size))
                trimmed_partition_indices = np.setdiff1d(partition_indices, saved_partition_indices)
                trimmed_demos[0] += [demos[0][idx] for idx in trimmed_partition_indices]
                trimmed_demos[1] += [demos[1][idx] for idx in trimmed_partition_indices]
                partition_indices = saved_partition_indices
            else:
                while partition_indices.shape[0] < partition_size:
                    partition_indices = np.tile(partition_indices, 2)
                partition_indices = partition_indices[:partition_size]
                
            partition = ([demos[0][idx] for idx in partition_indices],
                         [demos[1][idx] for idx in partition_indices])

            ## update partition
            partitions.append(partition)
            partition_indices_list.append(partition_indices)

        def assign_partition_fn(new_input_):
            """ Routing function """
            new_demo = self.temp_man.demos.fill(([new_input_], ['']))
            new_demo_embed = text_encoder.get_embedding(new_demo)
            new_demo_embed = new_demo_embed.mean(dim=(0, 1))
            data_matrix = new_demo_embed.numpy().reshape(1, -1)
            label = predict_fn(data_matrix).item()
            return label
        
        def get_closest_query_fn(expert_id, query_inputs):
            """ Used for sampling regional eval data (i.e. those assigned to each expert) """
            cluster_center = np.mean(data_matrix[partition_indices_list[expert_id]], axis=0).reshape(1, -1)
            query_input_embeds = []
            for query_input_ in (query_inputs):
                query_input = self.temp_man.demos.fill(([query_input_], ['']))
                query_input_embed = text_encoder.get_embedding(query_input)  # (1, seq_len, dim)
                query_input_embed = query_input_embed.mean(dim=(0, 1))
                query_input_embeds.append(query_input_embed)
            query_input_embeds = torch.stack(query_input_embeds)
            similarities = cosine_similarity(query_input_embeds.numpy(), cluster_center)
            closest_indices = np.argsort(similarities[:, 0])[::-1]
            return closest_indices

        self.cluster_assignments = cluster_assignments
        self.partition_indices_list = partition_indices_list
        return (partitions, assign_partition_fn, get_closest_query_fn), tuple(trimmed_demos)


    def _demos_to_embeds(self, demos, text_encoder):
        """ Convert demos ([inputs], [outputs]) to embeds (data matrix) """
        input_embeds = []
        for input_, _ in zip(*demos):
            input = self.temp_man.demos.fill(([input_], ['']))
            input_embed = text_encoder.get_embedding(input)  # (1, seq_len, dim)
            input_embed = input_embed.mean(dim=(0, 1))
            input_embeds.append(input_embed)
        inputs_embeds = torch.stack(input_embeds)
        data_matrix = inputs_embeds.numpy()
        return data_matrix
