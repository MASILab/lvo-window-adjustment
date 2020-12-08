import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv('csv/resplit_dataset.csv')
    df = df[df['set']=='val']
    x = df['window_level_manual']
    y = df['window_width_manual']
    z = df['lvo_radreport']
    plt.scatter(x, y, c=z, marker='o')
    plt.xlabel('window_level')
    plt.ylabel('window_width')
    plt.show()


if __name__ == '__main__':
    main()
