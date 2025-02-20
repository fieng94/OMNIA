import ollama
import pandas as pd
# Function to interract with llm

def row2text(row)-> tuple[str,str,str,str]:
    # Translate triple into text
    head = row['Head']
    relation = row['Relation']
    tail = row['Tail']
    prompt = f"Transform the following triple :({head},{relation},{tail}) into natural language text. Don't explain your choice"
    response = ollama.generate(model='mistral', 
            prompt= prompt)
    triple_text = response['response']
    return triple_text, head, relation, tail


def extract_score(text:str)->str:
    # Function to extract score
    # Find the starting index of "Score: "
    start_index = text.find("Score: ") + len("Score: ")

    # Find the end of the number, which could be marked by a non-digit character
    end_index = start_index
    while end_index < len(text) and (text[end_index].isdigit() or text[end_index] == '.'):
        end_index += 1

    # Extract and return the number using slicing
    return text[start_index:end_index]

def get_context_triple(df:pd.DataFrame, col_name:str, elem_comp:str, max_sample:int) -> list[str]:
    # get context triple
    context_df = df[df[col_name] == elem_comp]
    if len(context_df) > max_sample:
        context_df = context_df.sample(max_sample)
    list_triple = []
    for _,context_row in context_df.iterrows():
        head = context_row['Head']
        relation = context_row['Relation']
        tail = context_row['Tail']
        list_triple.append(f'({head},{relation},{tail})')
    return list_triple