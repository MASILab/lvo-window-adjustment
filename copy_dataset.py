import os
import pandas as pd

file = pd.read_csv("csv/resplit_dataset_backup.csv")
file['cta'] = file['cta'].apply(lambda x:
                                os.path.join('data', '/'.join(x.split('/')[5:])))

file.to_csv('csv/resplit_dataset.csv', index=False)

print(file['cta'])