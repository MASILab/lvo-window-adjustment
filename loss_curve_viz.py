import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('results/window_both_resplit_3fc_34_3e5_reg/epochs_summary.csv')
# df = pd.read_csv('results/window_both_resplit_3fc_reg/epochs_summary.csv')
# df = pd.read_csv('results/window_both_resplit_reg/epochs_summary.csv')
# df = pd.read_csv('results/window_both_reg/epochs_summary.csv')

# df = pd.read_csv('results/window_both_resplit_3fc_34_3e5_mt_reg/epochs_summary.csv')
df = pd.read_csv('results/window_both_resplit_3fc_mt_reg/epochs_summary.csv')

epochs = [int(i.split('_')[-1]) for i in df['epochs']]
train_loss = list(df['train_loss'])
val_loss = list(df['val_loss'])

plt.plot(epochs, train_loss, label='train_loss', marker='o')
plt.plot(epochs, val_loss, label='val_loss', marker='o')
plt.legend()
plt.title('train/val loss curve')
plt.xlabel('epochs')
plt.show()