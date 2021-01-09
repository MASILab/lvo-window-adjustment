import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('results/resnet_mt_2fc_reg_for_test/epochs_summary.csv')

    epochs = [int(i.split('_')[-1]) for i in df['epochs']]
    train_loss = list(df['train_loss'])
    val_loss = list(df['val_loss'])

    plt.plot(epochs, train_loss, label='train_loss', marker='o')
    plt.plot(epochs, val_loss, label='val_loss', marker='o')
    plt.legend()
    plt.title('train/val loss curve with lr=3e-4')
    plt.xlabel('epochs')
    plt.show()


if __name__ == '__main__':
    main()