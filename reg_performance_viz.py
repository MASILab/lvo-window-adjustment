import matplotlib.pyplot as plt
import os
import pandas as pd

level_df = pd.read_csv('results/window_level_reg/test.csv')
width_df = pd.read_csv('results/window_width_reg/test.csv')


level_pred = level_df['prediction']
width_pred = width_df['prediction']

level_target = level_df['target']
width_target = width_df['target']


def show_scatter_plot(level_pred, width_pred, level_target, width_target):
    plt.scatter(level_pred, width_pred, label='prediction', marker='o')
    plt.scatter(level_target, width_target, label='true_value', marker='o')
    plt.title('Regression performace')
    plt.xlabel('window_level')
    plt.ylabel('window_width')
    plt.legend()
    plt.show()
    plt.close()


show_scatter_plot(level_pred, width_pred, level_target, width_target)


def show_prediction_error(df, task):
    subj = df['subj']
    pred = df['prediction']
    tar = df['target']

    fig, ax = plt.subplots()
    ax.plot(pred, color='lightblue', marker='o', linestyle='none',
            markersize=6, label='Prediction')
    ax.plot(tar, color='red', marker='s', linestyle='none',
            markersize=6, label='Ground Truth')
    ax.tick_params(axis='x', rotation=90)
    ax.margins(0.05)
    ax.set(xticks=range(len(subj)), xticklabels=subj, ylabel=task)
    ax.legend(loc='best', numpoints=1)
    ax.grid(axis='x')
    plt.show()
    plt.close()


show_prediction_error(level_df, 'window_level')
show_prediction_error(width_df, 'window_width')
