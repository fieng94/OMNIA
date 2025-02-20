import pandas as pd
import experiment.preprocess as preprocess

path = 'data/codex/data.csv'
df = pd.read_csv(path)
# get the dataframe for evaluating our method
evaluation_df, candidates_df, missing_df = preprocess.create_experiment_df(path)

# filter the dataframe using TransE
model 

