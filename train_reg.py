import numpy as np
import pandas as pd
import nibabel as nib


def main():
    df = pd.read_csv('csv/dataset.csv')

    df_train = df[df['set'] == 'test']
    y_values = [[row['window_level_manual'], row['window_width_manual']] for index, row in df_train.iterrows()]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 2)
    print(y_train.shape)




if __name__ == '__main__':
    main()