import argparse
## DATA PREPARATION
import experiment.preprocess as preprocess
import experiment.filtering as filtering
import experiment.filtering as prep_llm
import numpy as np
import pandas as pd

path ='data/codex-m/data_sample.csv'


def main(path, setting="triples", subsetting='zero'):
    df = pd.read_csv(path)
    evaluation_df, candidates_df, missing_df = preprocess.create_experiment_df(path)
    filtred_df = filtering.create_filtred_df(df, evaluation_df, missing_df)

    filtred_df = filtred_df.merge(evaluation_df, how='left')
    filtred_df_sample = filtering.create_sample(filtred_df,sample_size= 500, true_cand_ratio= 0.5)

    if subsetting == 'rag':
        retriever = prep_llm.create_retriever(path, 2)
    if setting == 'triples':
        if subsetting == 'zero':
            score_list = prep_llm.plain_triple(filtred_df_sample)
        if subsetting == 'context':
            pass
        if subsetting == 'rag':
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OMNIA.")
    parser.add_argument("--name", type=str, default="World", help="Name of the person to greet")
    args = parser.parse_args()
    main(name=args.name, greeting=args.greeting)