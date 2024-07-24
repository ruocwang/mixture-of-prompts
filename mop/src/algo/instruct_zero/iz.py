import numpy as np
import torch
import logging

from algo.utils import generate, evaluate, data
from src.pe_manager import Prompter
from algo.automatic_prompt_engineer.demos_allocator import DemosAllocater
from src.exp_utils import load_args_from_dict, directories
from algo.instruct_zero.lm_forward_api import LMForwardAPI
from algo.instruct_zero.instruction_coupled_kernel import CombinedStringKernel, cma_es_concat

from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior


#### wrc comment: hardcoded parameters copied from IZ, rescaled to 20 evaluations for fair comparison
N_INIT = 11
N_ITERATIONS = 3
BATCH_SIZE = 3
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}


def find_prompts(data_man, temp_man, conf):
    args = conf['generation']['args']


    # Generate prompts (we shall just replace this with IZ)
    logging.info(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} prompts")
    #### args
    iz_args = load_args_from_dict(directories.iz_args)
    HF_cache_dir = iz_args.HF_cache_dir
    random_proj, intrinsic_dim, n_prompt_tokens = iz_args.random_proj, iz_args.intrinsic_dim, iz_args.n_prompt_tokens

    #### data
    prompt_gen_data = data_man.prompt_gen_data
    init_prompt = ['\n']
    prompt_gen_template = temp_man.gen
    d_template = temp_man.demos

    # when generating prompts, IZ only supports sampling one batch of pgen_data
    pgen_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'], seed=args.seed)
    init_qa = [prompt_gen_template.fill(d_template.fill(pgen_data))]

    #### model forward api
    def eval_fn(prompts):
        assert len(prompts) == 1
        prompters = [Prompter(prompts[0], conf, temp_man)]
        return evaluate.evaluate_prompts(prompters, data_man, conf['evaluation'], seed=args.seed)
    model_forward_api = LMForwardAPI(model_name=iz_args.model_name, init_prompt=init_prompt, 
                                    init_qa=init_qa, random_proj=random_proj, 
                                    intrinsic_dim=intrinsic_dim, n_prompt_tokens=n_prompt_tokens, 
                                    HF_cache_dir=HF_cache_dir, args=iz_args,
                                    eval_fn=eval_fn)

    #### BO init
    X = SobolEngine(dimension=intrinsic_dim, scramble=True, seed=0).draw(N_INIT)
    X_return = [model_forward_api.eval(x) for x in X]
    Y = [X[0] for X in X_return]
    Y_scores = [X[1].squeeze() for X in X_return]
    
    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
    Y_scores = torch.FloatTensor(np.array(Y_scores)).to(**tkwargs)
    print(f"Best initial point: {Y.max().item():.3f}")

    # standardization Y (no standardization for X)
    X_train = X
    y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2) + 1e-9)

    # define matern kernel
    matern_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=X_train.shape[-1],
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                )
    matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
    
    covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction, latent_train=X_train.double(), instruction_train=Y_scores))
    gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)



    #### Main search
    for i in range(N_ITERATIONS):
        print(f"X_train shape {X_train.shape}")
        print(f"y_train shape {y_train.shape}")

        fit_gpytorch_model(gp_mll)#, options = {'maxiter':10})
        EI = ExpectedImprovement(gp_model, best_f = y_train.max().item())
        
        starting_idxs = torch.argsort(-1*y_train.squeeze())[:BATCH_SIZE]
        starting_points = X_train[starting_idxs]

        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            if (torch.max(starting_point_for_cma) > 1 or torch.min(starting_point_for_cma) < -1):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs)
            best_points.append(newp)
            best_vals.append(newv)
            
        print(f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}")
        for idx in np.argsort(-1*np.array(best_vals)):
            X_next_point =  torch.from_numpy(best_points[idx]).float().unsqueeze(0)

            X_next_points_return = [model_forward_api.eval(X_next_point)]
            Y_next_point = [X[0] for X in X_next_points_return]
            Y_scores_next_points = [X[1].squeeze() for X in X_next_points_return]

            X_next_point = X_next_point.to(**tkwargs)
            Y_next_point = torch.FloatTensor(Y_next_point).unsqueeze(-1).to(**tkwargs)
            Y_scores_next_points = torch.FloatTensor(np.array(Y_scores_next_points)).to(**tkwargs)

            X = torch.cat([X, X_next_point])
            Y = torch.cat([Y, Y_next_point])
            Y_scores = torch.cat([Y_scores, Y_scores_next_points])

        # standardization Y
        X_train = X.clone()
        y_train = (Y - Y.mean(dim=-2))/(Y.std(dim=-2) + 1e-9)

        matern_kernel = MaternKernel(
                        nu=2.5,
                        ard_num_dims=X_train.shape[-1],
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    )
        matern_kernel_instruction = MaternKernel(
                nu=2.5,
                ard_num_dims=Y_scores.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction, latent_train=X_train.double(), instruction_train=Y_scores))
        gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        print(f"Best value found till now: {torch.max(Y)}")

    prompts = model_forward_api.return_best_prompt()
    prompters = [Prompter(prompt, conf, temp_man) for prompt in prompts]
    prompters = [prompters[0]]

    ## allocate demos
    demos_allocater = DemosAllocater(args.n_experts, conf, temp_man)
    prompters = demos_allocater.allocate_demos(data_man, prompters)

    ## evaluate
    if len(prompters) == 1:
        res = [evaluate.ExecAccuracyEvaluationResult(prompters, np.ones(1))]
    elif res is None:
        res = evaluate.evaluate_prompts(prompters, data_man, conf['evaluation'], seed=args.seed)

    return res

