import torch
import numpy as np
import copy
import sys
sys.path.insert(0, '.')
from src.evaluation.instruction_induction.exec_accuracy import exec_evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from automatic_prompt_engineer import evaluate, config, template
import re

from InstructZero.algo.instruct_zero.instruction_coupled_kernel import *

# from transformers.models.llama.modeling_llama import LlamaForCausalLM

# api_model = 'chatgpt'
api_model = 'llama'
alpha = 1
sigma = 1

class LMForwardAPI:
    def __init__(self, model_name='vicuna', eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, random_proj=None, intrinsic_dim=None, n_prompt_tokens=None, few_shot_data=None, 
                 HF_cache_dir=None):
        p = torch.ones(10)
        
        kwargs={'torch_dtype': torch.float16, "cache_dir": '/hf_cache'}
        if model_name in ["vicuna", "alpaca", "flan-t5"]:
            self.model = AutoModelForCausalLM.from_pretrained(
                                HF_cache_dir, low_cpu_mem_usage=True, **kwargs,
                            ).cuda()

            self.tokenizer = AutoTokenizer.from_pretrained(
                                HF_cache_dir,
                                model_max_length=512,
                                padding_side='left',
                                use_fast=False,
                            )
        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] + init_qa[0]
        if model_name in ['alpaca', 'vicuna']:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]

        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        
        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)
        
        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['llama', 'alpaca', 'vicuna']:
                print('Get the embedding firstly to avoid issues')
            else:
                raise NotImplementedError
            mu_hat = np.mean(self.embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(self.embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = alpha * std_hat / (np.sqrt(intrinsic_dim) * sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():   
                torch.nn.init.uniform_(p, -1, 1)
        elif random_proj == 'uniform':  
            for p in self.linear.parameters():   
                torch.nn.init.uniform_(p, -1, 1)

        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")
        
        if api_model in ['llama', 'flan-t5']:
            self.api_model = exec_evaluator(api_model, self.conf)

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data
        
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()

    def eval(self, prompt_embedding=None, test_data=None):
        """ Takes a compressed 1D soft prompt, output ? """
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
        
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            # if self.init_prompt is not None:
            #     prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        
        elif isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )

        input_ids = self.tokenizer(self.init_token, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)
        input_embed = torch.cat((prompt_embedding, input_embed), 1)  ## learned soft prefix + prompt (exemplers + The instruction was to...)
        outputs = self.model.generate(inputs_embeds=input_embed, max_new_tokens=64)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # postprocess instruction (extract only the first sentence of the instruction)
        instruction[0] = 'The instruction was to ' + instruction[0]
        start = instruction[0].find('The instruction was to')
        end = instruction[0].find('Comment:')
        if end == -1:
            instruction[0] = instruction[0][start:]
        else:
            instruction[0] = instruction[0][start: end]

        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', instruction[0])
        search_string = 'The instruction was to'
        for sentence in sentences:
            if sentence.startswith(search_string):
                instruction[0] = sentence
                break

        # print post-processed instruction
        print('Instruction: {}'.format(instruction))
        
        if instruction[0] in self.prompts_set.keys():  ## visited
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
        else:
            if api_model in ['chatgpt']: 
                dev_perf, instruction_score = evaluate.evaluate_prompts(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation']['method'], self.conf['evaluation'])
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            # We will fix the bugs for other api models. Stay tuned!
            elif api_model in ['llama', 'flan-t5']:
                dev_perf, instruction_score = self.api_model.evaluate(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data,
                                        self.conf['evaluation']).sorted()[1][0]            
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError

        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set

