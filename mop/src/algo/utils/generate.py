from . import data, llm
from src.pe_manager import Prompter


def get_query(temp_man, subsampled_data):
    """
    Returns a query for the prompt generator. A query is the prompt that is sent to the LLM.
    Parameters:
        subsampled_data: The data to use for the demonstrations.
    Returns:
        A query for the prompt generator.
    """
    inputs, outputs = subsampled_data
    demos = temp_man.demos.fill(subsampled_data)
    return temp_man.gen.fill(input=inputs[0], output=outputs[0], full_demo=demos)


def generate_prompts(prompt_gen_data, temp_man, conf):
    config = conf['generation']
    queries = []
    for i in range(config['num_subsamples']):
        subsampled_data = data.subsample_data(prompt_gen_data, config['num_demos'], seed=int(config['args'].seed*config['num_subsamples']+i))
        queries.append(get_query(temp_man, subsampled_data))

    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    prompts = model.generate_text(queries, n=config['num_prompts_per_subsample'])

    prompters = [Prompter(prompt, conf, temp_man) for prompt in prompts]

    return prompters
