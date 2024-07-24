import random
import torch
import numpy as np
import logging
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from algo.utils.models import GPT2, OpenAIModels, SentenceT5
from src.pe_manager import Prompter


class DemosAllocater():

    def __init__(self, n_experts, conf, temp_man):
        self.max_n_experts = n_experts
        self.conf = conf
        self.temp_man = temp_man

        self.make_prompter = lambda prompt: Prompter(prompt, conf, temp_man)


    def allocate_demos(self, data_man, prompters):
        self.demos_split  = self.assign_demos(data_man.few_shot_data)

        if self.demos_split == []:
            return prompters
        
        ## TODO under dev ## use add_demos()
        prompters_ = []
        for prompter in prompters:
            prompt = prompter.prompts[0]  + '\n\n' + self.temp_man.demos.fill(self.demos_split)
            prompters_.append(self.make_prompter(prompt))

        return prompters_


    def assign_demos(self, demos):
        demo_conf = self.conf['method']['demo']

        if self.max_n_experts == 1:
            demos_split = [demos]

        if demo_conf['algo'] == 'none':
            demos_split = []
        elif demo_conf['algo'] == 'fix':
            demos_split = self._fixed_partition(demos, self.max_n_experts)
        elif demo_conf['algo'] == 'kcen':
            demos_split = self._cluster_partition(demos, self.max_n_experts, demo_conf)
        else:
            raise ValueError(demo_conf['algo'])

        return demos_split


    #######################################################
    ################### demo assignment ###################
    #######################################################
    def _fixed_partition(self, demos, N): # ape + fixed: sample a fixed subset of data (a single partition)
        logging.info(f'[INFO] Fixed partition: {N} experts')

        # Calculate partition size
        partition_size = len(demos[0]) // N

        # Special case for length limit
        args = self.conf['generation']['args']
        if args.benchmark == 'bbh' and args.task == 'causal_judgement' and partition_size > 6:
            partition_size = 6

        # Shuffle indices
        shuffled_indices = list(range(len(demos[0])))
        random.shuffle(shuffled_indices)

        # Create partitions
        partitions = []
        partition_indices_list = []
        for i in range(N):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size
            partition_indices = shuffled_indices[start_idx:end_idx]
            partition_indices_list.append(partition_indices)
            
            partition = ([demos[0][idx] for idx in partition_indices],
                        [demos[1][idx] for idx in partition_indices])
            
            partitions.append(partition)

        partition = partitions[random.choice(range(N))]
        return partition


    def _cluster_partition(self, demos, N, demo_conf, prefix=''):
        """
            demos: ([inputs], [outputs])
        """
        logging.info(f'[INFO] Cluster partition: {N} experts')
        ## text embedding model
        if demo_conf['encoder'] == 'ada':
            text_encoder = OpenAIModels('text-embedding-ada-002')
        elif demo_conf['encoder'] == 't5':
            text_encoder = SentenceT5()
        else:
            raise ValueError(demo_conf['encoder'])

        ## turn tuple into icls
        data_matrix = self._demos_to_embeds(demos, text_encoder)

        #### clustering
        args = self.conf['generation']['args']

        kmeans = KMeans(n_clusters=min((len(demos[0]) // N), data_matrix.shape[0]), n_init='auto', random_state=args.seed)
        kmeans.fit(data_matrix)
        predict_fn = lambda data: kmeans.predict(data)
        cluster_assignments = kmeans.labels_

        cluster_centers = kmeans.cluster_centers_
        closest_indices = [np.argmin(cdist([center], data_matrix, 'cosine')) for center in cluster_centers]
        print(f'==> closest_indices = {closest_indices}...')

        partition_size = len(demos[0]) // N

        # Special case for length limit
        args = self.conf['generation']['args']
        if args.benchmark == 'bbh' and args.task == 'causal_judgement' and partition_size > 6:
            partition_size = 6

        if len(closest_indices) > partition_size:
            closest_indices = np.array(random.sample(list(closest_indices), partition_size))
        else:
            while len(closest_indices) < partition_size:
                closest_indices = np.tile(closest_indices, 2)
            closest_indices = closest_indices[:partition_size]

        partition = ([demos[0][idx] for idx in closest_indices],
                     [demos[1][idx] for idx in closest_indices])

        return partition


    def _demos_to_embeds(self, demos, text_encoder):
        """ convert demos ([inputs], [outputs]) to embeds (data matrix) """
        input_embeds = []
        for input_, _ in zip(*demos):
            input = self.temp_man.demos.fill(([input_], ['']))
            input_embed = text_encoder.get_embedding(input)  # (1, seq_len, dim)
            input_embed = input_embed.mean(dim=(0, 1))
            input_embeds.append(input_embed)
        inputs_embeds = torch.stack(input_embeds)
        data_matrix = inputs_embeds.numpy()
        return data_matrix

