import torch
import random
import numpy as np


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return f"Set all the seeds to {seed} successfully!"


#######################################################
################ Instruction Induction ################
#######################################################
TASKS=[
    'antonyms',
    'cause_and_effect',
    'common_concept',
    'diff',
    'first_word_letter',
    'informal_to_formal',
    'larger_animal',
    'letters_list',
    'taxonomy_animal',
    'negation',
    'num_to_verbal',
    'active_to_passive',
    'singular_to_plural',
    'rhymes',
    'second_word_letter',
    'sentence_similarity',
    'sentiment',
    'orthography_starts_with',
    'sum',
    'synonyms',
    'translation_en-de',
    'translation_en-es',
    'translation_en-fr',
    'word_in_context',
    'auto_categorization',
    'auto_debugging'
]


#######################################################
#################### Big-Bench-Hard ###################
#######################################################
BBH_TASKS = [
    'causal_judgement',
    'disambiguation_qa',
    'dyck_languages',
    'movie_recommendation',
    'navigate',
    'object_counting',
    'ruin_names',
    'snarks',
    'sports_understanding',
    'word_sorting'
]


# bbh tasks format
BBH_TASKS_FMT = {
    'causal_judgement': 'Y',
    'disambiguation_qa': 'M',
    'dyck_languages': 'S',
    'movie_recommendation': 'M',
    'navigate': 'Y',
    'object_counting': 'N',
    'ruin_names': 'M',
    'snarks': 'M',
    'sports_understanding': 'Y',
    'word_sorting': 'F'
}


BBH_FMT_PROMPT = {
    'N': 'The answer (arabic numerals) is',
    'M': 'Among ([FIRST]) through ([LAST]), the answer is',
    'Y': 'The answer (Yes or No) is',
    'T': 'The answer (True or False) is',
    'V': 'The answer (valid or invalid) is',
    'S': 'Your response should be a combination of the characters "<", ">", "{", "}", "[", "]", "(", ")".',
    'F': '',
}

#######################################################
###################### SuperNI ########################
#######################################################
SUPERNI_TASKS = [
    'task093_conala_normalize_lists',
    'task094_conala_calculate_mean',
    'task096_conala_list_index_subtraction',
    'task110_logic2text_sentence_generation',
    'task122_conala_list_index_addition',
    'task125_conala_pair_differences',
    'task921_code_x_glue_information_retreival',
    'task104_semeval_2019_task10_closed_vocabulary_mathematical_answer_generation',
    'task118_semeval_2019_task10_open_vocabulary_mathematical_answer_generation',
    'task750_aqua_multiple_choice_answering',
    'task834_mathdataset_classification',
    'task835_mathdataset_answer_generation',
    'task956_leetcode_420_strong_password_check',
    'task1419_mathqa_gain',
    'task1420_mathqa_general',
    'task1421_mathqa_other',
    'task1423_mathqa_geometry',
    'task1424_mathqa_probability',
    'task1678_mathqa_answer_selection',
    'task1726_mathqa_correct_answer_generation',
]
