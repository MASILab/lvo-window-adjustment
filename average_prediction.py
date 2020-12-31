import pandas as pd
from statistics import mean
import os

chunks_prediction = 'results/resnet_mt_2fc_reg_chunks/test.csv'

df = pd.read_csv(chunks_prediction, index_col=0)

subjects = list(set(df['subj']))
cols = ['prediction_level', 'prediction_width', 'target_level', 'target_width']
reduced_data = []
for subj in subjects:
    df_subj = df[df['subj'] == subj]
    averaged = [subj]
    for col in cols:
        averaged.append(mean(df_subj[col]))
    reduced_data.append(averaged)

df_reduced = pd.DataFrame(columns=df.columns, data=reduced_data)
df_reduced.to_csv(os.path.join(os.path.dirname(chunks_prediction), 'reduced_test.csv'))

