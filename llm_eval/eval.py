import ollama
import pandas as pd
from .utils import row2text, extract_score, get_context_triple
# Function to get context

# eval without context
def eval_text(triple_text):
    # give a score for a text
    prompt = f'We have the below sentence and you need to validate that and give a score to that.'
    prompt += f'given that a 0 is a score corresponding to invalide triple and 1 to valide triple give the corresponding score to this triple.'
    prompt += f"Is the following sentence :'{triple_text}' valid."
    prompt += f"Start your answer with :'Score:', if you can not determine if it is accurate give it a score of 0.5"
    response = ollama.generate(model='mistral', 
            prompt= prompt)
    text = response['response']
    score = extract_score(text)
    return score

def eval_df(df, size):
    # evaluate subsample of df with llm
    score_list = []
    idx = 0
    for index, row in df[:size].iterrows():
        triple_text, head, relation, tail = row2text(row)
        score = eval_text(triple_text)
        score_list.append(score)
        idx+= 1
        if idx%100 == 0:
            print(f'{idx}/{size}')
    print('Finished\n')
    return score_list


def eval_text_context(triple_text, context_head, context_rel, context_tail):
    # give a score for a text with context
    prompt = f'We have the below sentence and you need to validate that and give a score to that.'
    prompt += f'given that a 0 is a score corresponding to invalid sentence and 1 to valid sentence give the corresponding score to this triple.'
    prompt += f"Is the following sentence :'{triple_text}' valid."
    prompt += f"The following triples are example of triples that have the same head:{context_head}."
    prompt += f"The following triples are example of triples that have the same relation:{context_rel}."
    prompt += f"The following triples are example of triples that have the same tail:{context_tail}."
    prompt += f"Start your answer with :'Score:', if you can not determine if it is accurate give it a score of 0.5"
    response = ollama.generate(model='mistral', 
            prompt= prompt)
    text = response['response']
    score = extract_score(text)
    return score

def eval_df_context(df, size):
    # evaluate subsample of df with llm and context
    score_list = []
    idx = 0
    for index, row in df[:size].iterrows():
        
        triple_text, head, relation, tail = row2text(row)
        # get context for relations
        context_rel = get_context_triple(df, 'Relation', relation, 5)

        # get context for head
        context_head = get_context_triple(df, 'Head', head, 3)

        # get context for tail
        context_tail = get_context_triple(df, 'Tail', tail, 3)
        
        score = eval_text_context(triple_text, context_head, context_rel, context_tail)
        score_list.append(score)
        idx+= 1
        if idx%100 == 0:
            print(f'{idx}/{size}')
    print('Finished\n')
    return score_list
