import pandas as pd

def extract_unique_pair(df:pd.DataFrame, columns:list) -> pd.DataFrame:
    """Extracts unique pairs from a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to extract unique pairs from.
    columns : list
        List of columns to extract unique pairs from.
    
    Returns
    -------
    pd.DataFrame
        Dataframe of unique pairs.
    """
    df_unique = df.drop_duplicates(subset=columns).reset_index(drop=True)
    return df_unique

def extract_unique_rel_tail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a dataframe and return every pair of relations and tail in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to extract relations and tails from.

    Returns
    -------
    pd.DataFrame
        A dataframe containing every pair of relations and tails in the input dataframe.

    """
    columns = ['Relation', 'Tail']
    df_unique = extract_unique_pair(df, columns)
    df_unique = df_unique[columns]
    return df_unique
    
def extract_head_cluster(df: pd.DataFrame) -> list:
    """
    Extracts the head entities that share the same head entity for each unique pair of relation and tail in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the head, relation, and tail entities.

    Returns
    -------
    list
        A list of pandas dataframes, where each dataframe contains the head entities that share the same head entity for a unique pair of relation and tail.

    """
    # get every unique pair of relation and tail
    df_rel_tail = extract_unique_rel_tail(df)
    list_rel_tail = [row for index, row in df_rel_tail.iterrows()]
    # init head clusters
    list_head_cluster = []
    # for every pair, extract the head that share the same head
    for pair in list_rel_tail:
        cluster = pd.DataFrame()
        relation = pair['Relation']
        tail = pair['Tail']
        # condition to validate
        conditions = (df['Relation'] == relation) & (df['Tail'] == tail)
        cluster = df['Head'][conditions].drop_duplicates()
        if len(cluster) > 1:
            list_head_cluster.append(cluster)
    return list_head_cluster

def generate_combination_cluster(df: pd.DataFrame, cluster: pd.Series) -> pd.DataFrame:
    """
    Generate all possible combinations of entities in a given cluster.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the head, relation, and tail entities.
    cluster : pd.Series
        A series of entities that belong to the same cluster.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all possible combinations of entities in the given cluster.

    """
    rel_tail_cluster = df[["Relation", "Tail"]][df["Head"].isin(cluster)].drop_duplicates()
    head_series = cluster.repeat(len(rel_tail_cluster)).reset_index(drop=True)
    rel_tail_df = pd.concat([rel_tail_cluster] *len(cluster), ignore_index=True)
    combination_df = pd.concat([head_series, rel_tail_df], axis=1)
    return combination_df

def generate_all_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all possible combinations of entities in the input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the head, relation, and tail entities.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all possible combinations of entities in the input dataframe.

    """
    list_head_cluster = extract_head_cluster(df)
    candidates_df = pd.DataFrame()
    for cluster in list_head_cluster:
        combination_df = generate_combination_cluster(df, cluster)
        candidates_df = pd.concat([candidates_df, combination_df], axis=0)
    candidates_df = candidates_df.drop_duplicates()
    candidates_df = candidates_df.merge(df, how='left', indicator=True).query('_merge != "both"')
    candidates_df = candidates_df[['Head', 'Relation', 'Tail']]
    return candidates_df.reset_index(drop=True)