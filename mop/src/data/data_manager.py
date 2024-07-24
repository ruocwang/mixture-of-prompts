import random
import os
import sys
import json
sys.path.insert(0, '.')

## LM libraries
from algo.utils import data
from src.data.instruction_induction.load_data import load_data_ii
from src.data.bbh.load_data import load_data_bbh
from src.data.superni.load_data import load_data_superni
from src.data.other.load_data import load_data_other
from algo.utils import data


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _demos_to_embeds_notemplate(demos, text_encoder):
    import torch
    input_embeds = []
    for input_, _ in zip(*demos):
        input_embed = text_encoder.get_embedding(input_)  # (1, seq_len, dim)
        input_embed = input_embed.mean(dim=(0, 1))
        input_embeds.append(input_embed)
    inputs_embeds = torch.stack(input_embeds)
    data_matrix = inputs_embeds.numpy()
    return data_matrix

def cluster_data(demos):
    from k_means_constrained import KMeansConstrained
    from algo.utils.models import OpenAIModels
    text_encoder = OpenAIModels('text-embedding-ada-002')

    ## turn tuple into icls
    data_matrix = _demos_to_embeds_notemplate(demos, text_encoder)

    #### balanced kmean clustering
    size_min = data_matrix.shape[0] // 2 - 1
    size_max = data_matrix.shape[0] - size_min
    # kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0)
    # kmeans.fit(data_matrix)
    kmeans = KMeansConstrained(
        n_clusters=2,
        size_min=size_min,
        size_max=size_max,
        init='k-means++',
        random_state=0)
    kmeans.fit(data_matrix)
    cluster_assignments = kmeans.labels_

    induce_data, execute_data = ([], []), ([], [])
    for idx, cid in enumerate(cluster_assignments):
        if cid == 0:
            induce_data[0].append(demos[0][idx])
            induce_data[1].append(demos[1][idx])
        else:
            execute_data[0].append(demos[0][idx])
            execute_data[1].append(demos[1][idx])

    return induce_data, execute_data


class DataManager():
    """
    Instruction Induction Dataset.
        prompt_gen_data: The data to use for prompt generation.
        eval_data: The data to use for evaluation.
        few_shot_data: The data to use for demonstrations during eval (not implemented yet).
    """

    def __init__(self, benchmark, task):
        if benchmark == 'bbh':
            induce_data, test_data = load_data_bbh('induce', task), load_data_bbh('eval', task)
        elif benchmark == 'superni':
            induce_data, test_data = load_data_superni('induce', task), load_data_superni('eval', task)
        elif benchmark == 'ii':
            induce_data, test_data = load_data_ii('induce', task), load_data_ii('eval', task)
        else:
            raise ValueError('Benchmark not found!')
        
        #### split dataset
        induce_data_size = len(induce_data[0])
        prompt_gen_size = min(int(induce_data_size * 0.5), 100)
        eval_size = prompt_gen_size * 4

        prompt_gen_data, eval_data = data.create_split(induce_data, prompt_gen_size)
        if benchmark == 'bbh':
            print("Using 20 pgen data for BBH to avoid exceeding max token limit.")
            prompt_gen_data = (prompt_gen_data[0][:20], prompt_gen_data[1][:20])

        eval_data = (eval_data[0][:eval_size], eval_data[1][:eval_size])
        # Data is in the form input: single item, output: list of items
        # For prompt_gen_data, sample a single item from the output list
        prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0] for output in prompt_gen_data[1]]


        #### properties
        self.prompt_gen_data = prompt_gen_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.few_shot_data = prompt_gen_data  ## prompt_gen_data is the only slice used for training

        self.prompt_gen_data_size = len(prompt_gen_data[0])
        self.eval_data_size = len(eval_data[0])
        self.test_data_size = len(test_data[0])
        self.few_shot_data_size = len(self.few_shot_data[0])

    def get_split(self, split):
        return getattr(self, f'{split}_data')
