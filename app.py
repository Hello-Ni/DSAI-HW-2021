
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import model
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
# You can write code above the if-main block.
num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1


def createSequence(data, seq_len):

    # create sequence data,x is previous time step,y is next time period

    xs = []
    ys = []
    for i in range(len(data)-seq_len-1):
        x = data[i:(i+seq_len)]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    energe_data = pd.read_csv(args.training)
    energe_data = energe_data.iloc[:, 3:4].values

    scaler = MinMaxScaler()
    # change data range to [0~1]
    energe_data = scaler.fit_transform(
        energe_data)
    seq_len = 4
    x, y = createSequence(energe_data, seq_len)
    n_train = int(len(y) * 0.67)
    data_x = Variable(torch.Tensor(np.array(x)))
    data_y = Variable(torch.Tensor(np.array(y)))

    train_x = Variable(torch.Tensor(np.array(x[:n_train])))
    train_y = Variable(torch.Tensor(np.array(y[:n_train])))

    testX = Variable(torch.Tensor(np.array(x[n_train:])))
    testY = Variable(torch.Tensor(np.array(y[n_train:])))
    # plt.plot(energe_data, label='data')
    # plt.show()

    lstm = model.LSTM_Model(num_classes, input_size,
                            hidden_size, num_layers, seq_len)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(train_x)
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, train_y)

        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    lstm.eval()
    train_predict = lstm(data_x)
    data_predict = train_predict.data.numpy()
    data_y_plot = data_y.data.numpy()

    data_predict = scaler.inverse_transform(data_predict)
    data_y_plot = scaler.inverse_transform(data_y_plot)

    plt.axvline(x=n_train, c='r', linestyle='--')

    plt.plot(data_y_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    # plt.show()
    # predict result 3/30~4/10

    recent_data = data_x[-1:, :, :]
    future_data = []
    for i in range(14):
        next_data = lstm(recent_data)
        recent_data = torch.cat((recent_data, next_data.unsqueeze(2)), 1)
        recent_data = recent_data[:, 1:, :]

        next_arr = next_data.data.numpy().flatten().tolist()
        future_data.append(next_arr)
    future_data = np.array(future_data)
    future_data_plot = scaler.inverse_transform(future_data).flatten().tolist()
    begin_date = 20220401
    date = [20220330, 20220331]
    for i in range(1, 13):
        date.append((begin_date+i))

    date = [str(i) for i in date]
    future_data = [int(i) for i in future_data]
    result = pd.DataFrame(
        {'date': date, 'operating_reverse(MW)': future_data_plot})
    result.to_csv("submission.csv")
