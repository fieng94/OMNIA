import pandas as pd
from candidates_filtering.embedding import train_model
from candidates_filtering.embedding.get_emb_transe import get_list_dist
from candidates_filtering import triple_filter

def train_transe_embedding(df:pd.DataFrame):
    """
    Train TransE embedding on a dataframe and return the training df with the embedding model
    """
    train_df = train_model.create_dataset(
        df)
    test_df = train_model.create_dataset(
        df.sample(n=50))
    embedding_dim = 5
    model_kwargs = {"embedding_dim": embedding_dim}

    model_name = 'TransE'
    experiment_name = model_name+f"_dim{embedding_dim}"
    model = train_model.create_pipeline(train_df, test_df,
                        model_name, model_kwargs, experiment_name)
    return model, train_df

def get_filtred_df(model, candidates_df:pd.DataFrame, missing_df:pd.DataFrame, train_df:pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe using the best threshold for TransE score
    """
    list_dist = get_list_dist(candidates_df,model.model, train_df)
    candidates_df['distance'] = list_dist
    _, filtred_df = triple_filter.filter_best_threshold(model, candidates_df, missing_df, train_df)
    return filtred_df

def create_filtred_df(df:pd.DataFrame, candidates_df:pd.DataFrame, missing_df:pd.DataFrame) -> pd.DataFrame:
    """
    Creating filtred dataframe dataframe using the best threshold for TransE score
    """
    model, train_df = train_transe_embedding(df)
    filtred_df = get_filtred_df(model,candidates_df, missing_df, train_df)
    return filtred_df

def create_sample(df:pd.DataFrame, sample_size:int=500, true_cand_ratio:float=0.5) -> pd.DataFrame:
    """
    Creating sample of the dataframe
    """
    true_cand_size = int(sample_size * true_cand_ratio)
    false_cand_size = sample_size - true_cand_size
    true_cand_df = df[df['Missing'] == 1].sample(true_cand_size)
    false_cand_df = df[df['Missing'] == 0].sample(false_cand_size)
    df_sample = pd.concat([true_cand_df, false_cand_df])
    df_sample = df_sample.sample(frac=1)
    return df_sample