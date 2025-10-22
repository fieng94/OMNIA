import argparse
## DATA PREPARATION
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

def main(data_path, cand_path, output_dir, setting="triples", subsetting='zero', top_k=2):
    # Checking if arg are correct
    start = time.perf_counter()
    assert setting in ['triples', 'sentences'], f"{setting} does not exist as setting!"
    assert subsetting in ['zero','context','rag'], f"{subsetting} does not exist as subsetting!"
    # Read data
    print(f'Reading data at {data_path}')
    df = pd.read_csv(data_path)
    # Generate missing data candidates
    print(f'Reading the candidate to evaluate at {cand_path}')
    filtred_df_sample = pd.read_csv(cand_path)
    # Candidate validation using LLM
    print('Evaluation of missing data candidate')
    if subsetting == 'rag':
        retriever = prep_llm.create_retriever(data_path, top_k)
    if setting == 'triples':
        if subsetting == 'zero':
            score_list = prep_llm.plain_triple(filtred_df_sample)
        elif subsetting == 'context':
            score_list = prep_llm.context_triple(filtred_df_sample, df)
        elif subsetting == 'rag':
            score_list = prep_llm.RAG_triple(filtred_df_sample, retriever)
    elif setting == 'sentences':
        if subsetting == 'zero':
            score_list = prep_llm.plain_sentence(filtred_df_sample)
        elif subsetting == 'context':
            score_list = prep_llm.context_sentence(filtred_df_sample, df)
        elif subsetting == 'rag':
            score_list = prep_llm.RAG_sentence(filtred_df_sample, retriever)
    # Result extraction
    print('Finished Evaluation')
    print(f'Writing output in {output_dir}')
    prediction, ground_truth = result.get_gt_pred(filtred_df_sample, score_list)
    accuracy, f1_score, recall, precision = result.compute_score(prediction, ground_truth)
    # output evaluated sample
    output_eval_df_path = os.path.join(output_dir,f'{setting}_{subsetting}_evaluated_df.csv')
    filtred_df_sample.to_csv(output_eval_df_path)
    end = time.perf_counter()
    # output score
    res = [accuracy,precision,recall,f1_score,end-start]
    result_df =  pd.DataFrame([res], columns=["Accuracy", "Precision", "Recall", "F1 score","Execution time"])
    output_res_df_path = os.path.join(output_dir,f'{setting}_{subsetting}_results.csv')
  
    print(f"Execution time: {end - start:.2f} seconds")
    result_df.to_csv(output_res_df_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OMNIA.")
    parser.add_argument("--data_path", type=str, help="Path to evaluated KG in CSV format")
    parser.add_argument("--cand_path", type=str, help="Path to candidates triples in CSV format")
    parser.add_argument("--output_dir", type=str, help='Path to where the result will be sent')
    parser.add_argument("--setting", type=str, default='triples',
                        help="Evaluation on plain triples or transforming into sentences.\
                        Use arg 'triples' for triples or 'sentences' for sentence")
    parser.add_argument("--subsetting", type=str, default='zero',
                        help="Evaluation using zero-shot, in-context or RAG.\
                            Use arg 'zero' for zero-shot, 'context' for in-context and 'rag' for RAG")
    parser.add_argument("--top_k", type=int, default=2,
                    help="top-k to use for RAG.")
    args = parser.parse_args()
    main(data_path=args.data_path,
         cand_path=args.cand_path, 
         output_dir=args.output_dir,
         setting=args.setting, 
         subsetting=args.subsetting,
         top_k = args.top_k)