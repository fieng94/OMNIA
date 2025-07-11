import pandas as pd 
import experiment.preprocess as preprocess
## KGE FILTERING
import experiment.filtering as filtering
## LLM EVALUATION
import experiment.prep_llm as prep_llm
import experiment.result as result
import numpy as np
import pandas as pd
import os

def gen_cand(dataset:str):
    path =  f'data/{dataset}/data.csv'
    # Read data
    print(f'Reading data at {path}')
    df = pd.read_csv(path)
    # Generate missing data candidates
    print('Generating missing data candidates')
    evaluation_df, candidates_df, missing_df = preprocess.create_experiment_df(path)   
    return df, evaluation_df, candidates_df, missing_df

def filtrering_df(df, evaluation_df, missing_df):
    print('Filtering using Knowledge Graph Embedding (TransE)')
    filtred_df = filtering.create_filtred_df(df, evaluation_df, missing_df)
    filtred_df = filtred_df.merge(evaluation_df, how='left')
    filtred_df_sample = filtering.create_sample(filtred_df,sample_size= 500, true_cand_ratio= 0.5)
    return filtred_df, filtred_df_sample

dataset_list = ['freebase','wordnet','codex-m']
for dataset in dataset_list:
    df, evaluation_df, candidates_df, missing_df = gen_cand(dataset)
    filtred_df, filtred_df_sample = filtrering_df(df, evaluation_df, missing_df)
    cand_path = f'data/{dataset}/cand_sample.csv'
    filtred_df_sample.to_csv(cand_path, index=None)