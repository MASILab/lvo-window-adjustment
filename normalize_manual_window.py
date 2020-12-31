import pandas as pd


df = pd.read_csv('csv/resplit_dataset_no_outlier.csv')

df_train = df[df['set'] == 'train']

level_train = list(df_train['window_level_manual'])
width_train = list(df_train['window_width_manual'])


def min_range(data):
    data_min = min(data)
    data_max = max(data)
    data_range = data_max - data_min

    return data_min, data_range


level_min, level_range = min_range(level_train)
print(level_min, level_range)
width_min, width_range = min_range(width_train)
print(width_min, width_range)


def normalize(x, min, range):
    return (x - min)/range


df['window_level_manual'] = df['window_level_manual'].apply(lambda x: (x - level_min)/level_range)
df['window_width_manual'] = df['window_width_manual'].apply(lambda x: (x - width_min)/width_range)

df.to_csv('csv/resplit_norm_dataset_no_outlier.csv', index=0)