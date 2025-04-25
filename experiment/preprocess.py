import pandas as pd
from candidates_generation import triple_gen

def compute_missing_df(original_df:pd.DataFrame, sample_df:pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute data that are supposed to be recovered, 
    i.e. that exist in the original dataset but in the sample
    """
    df_missing = original_df[~original_df.apply(tuple, axis=1).isin(sample_df.apply(tuple, axis=1))]
    return df_missing

def create_ground_truth(candidates_df:pd.DataFrame, missing_df:pd.DataFrame, nb_true_cand:int=0, nb_false_cand:int=0) -> pd.DataFrame:
    """
    Function to compute the ground truth of the generated candidates.
    i.e. the generated candidates that exist in the original data is positive
    the generated candidates that doesn't exist in the original data is negative
    """
    true_cand_df = candidates_df[candidates_df.apply(tuple, axis=1).isin(missing_df.apply(tuple, axis=1))]
    false_cand_df = candidates_df[~candidates_df.apply(tuple, axis=1).isin(missing_df.apply(tuple, axis=1))]
    # by default the number of positive and negative sample will be 1000 unless it contains less than 1000 triples
    if nb_true_cand == 0:
        nb_true_cand = min(1000,len(true_cand_df))
    if nb_false_cand == 0:
        nb_false_cand = min(1000,len(false_cand_df))
    # if  there is negative number of candidates
    assert nb_true_cand > 0, f'This program need a positive number of true candidates, you chose {nb_true_cand}.'
    assert nb_false_cand > 0, f'This program need a positive number of false candidates, you chose {nb_false_cand}.'
    # get the testing data
    true_cand_df['Missing'] = 1
    false_cand_df['Missing'] = 0
    evaluation_df = pd.concat([true_cand_df.sample(nb_true_cand), false_cand_df.sample(nb_false_cand)])
    evaluation_df = evaluation_df.sample(frac = 1)
    return evaluation_df

def create_experiment_df(path:str, sample_proportion:float=0.8, nb_true_cand:int=0, nb_false_cand:int=0) -> tuple:
    """
    Read a csv file with triples, generated the candidates and the ground truth
    return in that order :  evaluation df   : to evaluate the method
                            candidates df   : the generated candidates
                            missing df      : the data that we want to recover
    """
    # read dataset
    df = pd.read_csv(path)
    # apply our method to only x % of the dataset to try to recover the others
    sample_size = int(sample_proportion * len(df))
    sample_df = df.sample(int(sample_size))
    # get missing data that need to be recovered
    missing_df = compute_missing_df(df, sample_df)
    # get candidates that are generated
    candidates_df = triple_gen.generate_all_candidates(sample_df)
    # get the evaluation df with ground truth
    evaluation_df = create_ground_truth(candidates_df, missing_df, nb_true_cand, nb_false_cand)
    return evaluation_df, candidates_df, missing_df
