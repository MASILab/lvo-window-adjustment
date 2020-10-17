import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from scipy.stats import ks_2samp

def main():
    df = pd.read_csv('csv/dataset.csv')

    subj_level_dict = pd.Series(df['window_level_manual'].values, index=df['subj']).to_dict()
    level = list(subj_level_dict.values())
    subj = list(subj_level_dict.keys())

    interval = 30
    bins = []
    for i in range(min(level), max(level)+1, interval):
        bins.append(i)

    random.seed(120)
    subj.sort()
    random.shuffle(subj)

    # split train-val-test set
    train_split = int(len(subj)*0.6)
    train_list = subj[:train_split]

    val_split = int(len(subj)*0.8)
    val_list = subj[train_split:val_split]

    test_list = subj[val_split:]

    # find the corresponding window level to check the quality of the split
    train_level = [subj_level_dict[i] for i in train_list]
    val_level = [subj_level_dict[i] for i in val_list]
    test_level = [subj_level_dict[i] for i in test_list]

    train_val_check = ks_2samp(train_level, val_level)
    val_test_check = ks_2samp(train_level, test_level)
    train_test_check = ks_2samp(val_level, test_level)

    if train_val_check.pvalue > 0.05 and val_test_check.pvalue > 0.05 and train_test_check.pvalue > 0.05:
        print('Successful split, pvalue: {}, {}, {}'.
              format(train_val_check.pvalue, val_test_check.pvalue, train_test_check.pvalue))
    else:
        print('Unsuccessful split! pvalue: {}, {}, {}'.
              format(train_val_check.pvalue, val_test_check.pvalue, train_test_check.pvalue))
        exit()

    # create a dictionary with subj as index and split as value
    train_split_dict = split_dict(train_list, 'train')
    val_split_dict = split_dict(val_list, 'val')
    test_split_dict = split_dict(test_list, 'test')
    data_split_dict = {**train_split_dict, **val_split_dict, **test_split_dict}

    # create a df with re-splitted train-val-test set
    df_resplit = df.copy()
    resplit_set = []
    for idx, row in df_resplit.iterrows():
        resplit_set.append(data_split_dict[row['subj']])

    df_resplit['set'] = resplit_set
    df_resplit.to_csv('csv/resplit_dataset.csv')

    # visualize the distribution within each split
    viz_dist(train_level, bins, 1)
    viz_dist(val_level, bins, 2)
    viz_dist(test_level, bins, 3)

    plt.show()


def viz_dist(data, bins, num):
    plt.figure(num)
    arr = plt.hist(data, bins=bins, edgecolor='k')
    plt.xticks(bins)
    plt.ylim(0, 50)
    for i in range(len(bins)-1):
        plt.text((arr[1][i]), arr[0][i], str(arr[0][i]))


def split_dict(subj_list, split):
    data_dict = {}
    for i in subj_list:
        data_dict[i] = split
    return data_dict


if __name__ == '__main__':
    main()

