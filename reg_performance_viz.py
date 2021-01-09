import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/resnet_mt_2fc_reg_for_test_1/test.csv')

def show_scatter_plot(level_pred, width_pred, level_target, width_target):
    plt.scatter(level_pred, width_pred, label='prediction', marker='o')
    plt.scatter(level_target, width_target, label='true_value', marker='o')
    plt.title('Regression performace')
    plt.xlabel('window_level')
    plt.ylabel('window_width')
    plt.legend()
    plt.show()
    plt.close()


level_pred = df['prediction_level']
width_pred = df['prediction_width']

level_target = df['target_level']
width_target = df['target_width']

show_scatter_plot(level_pred, width_pred, level_target, width_target)


def show_prediction_error(df, task):
    subj = df['subj']
    if task == 'window_level':
        pred = df['prediction_level']
        tar = df['target_level']
    else:
        pred = df['prediction_width']
        tar = df['target_width']

    data_tuple_list = [(subj, pred, tar) for subj, pred, tar in zip(subj, pred, tar)]
    data_tuple_list.sort(key=lambda x: x[2])
    subj = [i[0] for i in data_tuple_list]
    pred = [i[1] for i in data_tuple_list]
    tar = [i[2] for i in data_tuple_list]

    fig, ax = plt.subplots()
    ax.plot(pred, color='lightblue', marker='o', linestyle='none',
            markersize=5, label='Prediction')
    ax.plot(tar, color='red', marker='s', linestyle='none',
            markersize=3, label='Ground Truth')
    ax.tick_params(axis='x', rotation=90)
    ax.margins(0.05)
    ax.set(xticks=range(len(subj)), xticklabels=subj, ylabel=task)
    ax.legend(loc='best', numpoints=1)
    ax.grid(axis='x')
    plt.show()
    plt.close()


show_prediction_error(df, 'window_level')
show_prediction_error(df, 'window_width')
