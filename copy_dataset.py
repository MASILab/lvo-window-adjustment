import os
import pandas as pd

file = pd.read_csv("csv/resplit_dataset.csv", index_col=0)
file['mip'] = file['mip'].apply(lambda x:
                                os.path.join('data', '/'.join(x.split('/')[5:])))

file.to_csv('csv/resplit_dataset.csv')

print(file['cta'])
