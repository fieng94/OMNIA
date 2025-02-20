import pandas as pd
from candidates_filtering.embedding.get_emb_transe import get_list_dist

def filter_candidates(candidates_df:pd.DataFrame, threshold:float) -> pd.DataFrame:
    """
    Filter out candidates below a certain threshold
    """
    candidates_sample_df = candidates_df[candidates_df['distance']<threshold]
    candidates_sample_df = candidates_sample_df[['Head','Relation','Tail']]
    return candidates_sample_df

def compute_coverage(filtred_df :pd.DataFrame, missing_df:pd.DataFrame) -> float:
    """
    Compute how much of the missing data we are able to recover from the filtred candidates
    """
    coverage_df = filtred_df[filtred_df.apply(tuple, axis=1).isin(missing_df.apply(tuple, axis=1))]
    coverage = len(coverage_df) / len(missing_df)
    return coverage

def filter_best_threshold(model, candidates_df:pd.DataFrame, missing_df:pd.DataFrame,
                          train_df:pd.DataFrame, threshold_list:list = None) -> pd.DataFrame:
    """
    Compute the best threshold out of a list of threshold for a model.
    Return the best threshold and the filtred triples.
    """
    score_dict = {}
    # default threshold list based on mean distance
    if threshold_list == None:
        list_dist = get_list_dist(candidates_df, model.model, train_df)
        candidates_df['distance'] = list_dist
        mean_dist = candidates_df['distance'].mean()
        std_dist = candidates_df['distance'].std()
        threshold_list = [mean_dist- std_dist, mean_dist-0.5*std_dist,mean_dist,
                        mean_dist+0.5*std_dist,mean_dist + std_dist]
        
    for threshold in threshold_list:
        filtred_df = filter_candidates(candidates_df, threshold)
        coverage = compute_coverage(filtred_df, missing_df)
        reduction_ratio = len(filtred_df)/len(candidates_df)
        new_score = coverage/reduction_ratio
        score_dict[threshold] = new_score, reduction_ratio, coverage

    rr_cov_list = [(1- score_dict[key][1])*score_dict[key][2] for key in score_dict]
    index_threshold = rr_cov_list.index(max(rr_cov_list))
    threshold = threshold_list[index_threshold]
    filtred_df = filter_candidates(candidates_df, threshold)
    return threshold, filtred_df
    