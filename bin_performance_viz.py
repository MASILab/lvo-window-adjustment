import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd

acc_history = []
prec_history = []
rec_history = []
f1_history = []

for i in range(20):
    fname = 'epoch_' + str(i) + '.csv'
    df = pd.read_csv(os.path.join('epochs',fname))
    y_pred = df['prediction']
    y_target = df['target']
    acc = accuracy_score(y_target, y_pred)
    acc_history.append(acc)

    prec = precision_score(y_target, y_pred)
    prec_history.append(prec)

    rec = recall_score(y_target, y_pred)
    rec_history.append(rec)

    f1 = f1_score(y_target, y_pred)
    f1_history.append(f1)


epochs = [i for i in range(20)]

plt.plot(epochs, acc_history, label='accuracy', marker='o')
plt.plot(epochs, f1_history, label='f1_score', marker='o')
# plt.plot(epochs, prec_history, label='precision', marker='o')
# plt.plot(epochs, rec_history, label='recall', marker='o')
plt.title('binned performace over epochs')
plt.xlabel('epochs')
plt.grid(True)
plt.legend()
plt.show()