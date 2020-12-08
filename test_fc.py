import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


def main():
    x_values = torch.linspace(0, 100, 100)
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [-i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    y1_values = [i**3 for i in x_values]
    y1_train = np.array(y1_values, dtype=np.float32)
    y1_train = y1_train.reshape(-1, 1)

    input_dim = 1
    output_dim = 1
    lr = 0.001
    epochs = 4000
    model = LinearRegression(input_dim, output_dim)

    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
            labels_1 = Variable(torch.from_numpy(y_train).cuda())
            labels_2 = Variable(torch.from_numpy(y1_train).cuda())
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels_1 = Variable(torch.from_numpy(y_train))
            labels_2 = Variable(torch.from_numpy(y1_train))

        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs[0], labels_1) + criterion(outputs[1], labels_2)
        print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        # print('epoch {}, loss {}'.format(epoch, loss.item()))

    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted_1 = model(Variable(torch.from_numpy(x_train).cuda()))[0].cpu().data.numpy()
            predicted_2 = model(Variable(torch.from_numpy(x_train).cuda()))[1].cpu().data.numpy()
        else:
            predicted_1 = model(Variable(torch.from_numpy(x_train)))[0].data.numpy()
            predicted_2 = model(Variable(torch.from_numpy(x_train)))[1].data.numpy()
        print(predicted_1, predicted_2)

    plt.clf()
    #plt.plot(x_train, y_train, 'go', label='True data', alpha=0.3)
    #plt.plot(x_train, predicted_1, '--', label='Predictions', color='red',  alpha=0.8)
    plt.plot(x_train, y1_train, 'go', label='True data', alpha=0.3)
    plt.plot(x_train, predicted_2, '--', label='Predictions', color='red', alpha=0.8)
    plt.legend(loc='best')
    plt.show()

    print()


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(in_features=input_size, out_features=100),
                                    nn.ReLU(),
                                    nn.Linear(in_features=100, out_features=output_size))

        self.linear2 = nn.Sequential(nn.Linear(in_features=input_size, out_features=100),
                                     nn.ReLU(),
                                     nn.Linear(in_features=100, out_features=output_size))

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.linear2(x)
        return y1, y2


if __name__ == '__main__':
    main()
