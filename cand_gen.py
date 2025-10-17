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
import time

def gen_cand(dataset:str):
    path =  f'data/{dataset}/data_sample.csv'
    # Read data
    print(f'Reading data at {path}')
    df = pd.read_csv(path)
    # Generate missing data candidates
    print('Generating missing data candidates')
    evaluation_df, candidates_df, missing_df = preprocess.create_experiment_df(path)   
    return df, evaluation_df, candidates_df, missing_df

def filtrering_df(df, evaluation_df, missing_df, sample_size):
    print('Filtering using Knowledge Graph Embedding (TransE)')
    filtred_df = filtering.create_filtred_df(df, evaluation_df, missing_df)
    filtred_df = filtred_df.merge(evaluation_df, how='left')
    filtred_df_sample = filtering.create_sample(filtred_df,sample_size= sample_size, true_cand_ratio= 0.5)
    return filtred_df, filtred_df_sample

records = []

dataset_list = ['freebase','wordnet','codex-m']
sample_size_list = [500,1000,1500]
for dataset in dataset_list:
        for sample_size in sample_size_list:
            start = time.perf_counter()
            df, evaluation_df, candidates_df, missing_df = gen_cand(dataset)
            filtred_df, filtred_df_sample = filtrering_df(df, evaluation_df, missing_df, sample_size)
            cand_path = f'data/{dataset}/cand_sample_{sample_size}.csv'
            filtred_df_sample.to_csv(cand_path, index=None)
            end = time.perf_counter()
            print(f"Execution time: {end - start:.6f} seconds\tDataset:{dataset}\tSample size:{sample_size}")
            records.append({
            "dataset": dataset,
            "sample_size": sample_size,
            "execution_time_sec": end - start
            })

# Convert to DataFrame
time_df = pd.DataFrame(records)

# Save as CSV for later analysis
time_df.to_csv("timing_results.csv", index=False)