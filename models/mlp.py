import torch.nn as nn
import torch.nn.functional as F
import torch

# define NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        # linear layer (784 -> hidden_1)
        self.input = nn.Linear(512 * 512, 512)

        self.hidden = nn.Linear(512, 256)

        self.task_1 = nn.Sequential(nn.Linear(256, 128),
                                    nn.Linear(128, 1))
        self.task_2 = nn.Sequential(nn.Linear(256, 128),
                                    nn.Linear(128, 1))
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 512 * 512)
        # add hidden layer, with relu activation function
        x = self.input(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.hidden(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x1 = self.task_1(x)
        x2 = self.task_2(x)
        x_ = torch.cat((x1.T, x2.T), 0)

        return x_