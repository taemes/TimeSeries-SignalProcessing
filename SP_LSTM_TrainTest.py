
# https://coding-yoon.tistory.com/131 #

import os
import math
import time
import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler

from bayes_opt import BayesianOptimization

start = time.time()

# File Load #
path_raw = './data_raw/'
# path_test = './data_raw/mov_test/finger_test'
path_test = './data_raw/mov_test/'
path_out = './out_lstm_raw_test'
file_train = 'data_train_finger.csv'
file_test = '21.12.21_17h48m_3.csv'

Train = os.path.join(path_raw, file_train)
Test = os.path.join(path_test, file_test)
dataTrain = np.loadtxt(Train, delimiter=',', dtype=np.float32)
dataTest = np.loadtxt(Test, delimiter=',', dtype=np.float32)


# x_raw = data[:, 0:-1]
x_raw = dataTrain[:, [0]]
y_raw = dataTrain[:, [-1]]

x_test = dataTest[:, [0]]
y_test = dataTest[:, [-1]]
window_size = 25


def moving_average(data_raw, window_size):
    data_raw = np.array(data_raw).reshape(-1)
    data_raw = pd.DataFrame(data_raw)

    mov_avg = data_raw.rolling(window_size, min_periods=1).mean()
    mov_out = data_raw - mov_avg
    return mov_out


XTrain = moving_average(x_raw, window_size)
YTrain = moving_average(y_raw, window_size)

XTest = moving_average(x_test, window_size)
YTest = moving_average(y_test, window_size)


# 데이터 정규화 #
# StandardScaler  각 특징의 평규을 0, 분산을 1이 되도록 변경
# MinMaxScaler   최대/최소 값이 각각 1, 0이 되도록 변경
# RobustScaler

ss = StandardScaler()
mm = RobustScaler()

x_ss = ss.fit_transform(XTrain)
y_mm = mm.fit_transform(YTrain)


print("Training Shape", XTrain.shape, YTrain.shape)
print("Testing Shape", XTest.shape, YTest.shape)

XTrain_tensors = Variable(torch.Tensor(x_ss))
# XTest_tensors = Variable(torch.Tensor(XTest))

YTrain_tensors = Variable(torch.Tensor(y_mm))
# YTest_tensors = Variable(torch.Tensor(YTest))

XTrain_tensors_final = torch.reshape(XTrain_tensors,
                                     (XTrain_tensors.shape[0], 1, XTrain_tensors.shape[1]))
# XTest_tensors_final = torch.reshape(XTest_tensors,
#                                     (XTest_tensors.shape[0], 1, XTest_tensors.shape[1]))

# print("Training Shape", XTrain_tensors_final.shape, YTrain_tensors.shape)
# print("Testing Shape", XTest_tensors_final.shape, YTest_tensors.shape)


# device 선정  #
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


# 네트워크 파라미터 구성 #

def set_bs():
    bs = 2048
    if file_train == 'data_train_finger.csv':
        bs = 8192
    return bs


batch_size = set_bs()
# batch_size = len(XTrain)
num_epochs = 2000
learning_rate = 1e-5


input_size = 1  # number of features
hidden_size = 4  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers
num_classes = 1  # number of output classes

dataset = TensorDataset(XTrain_tensors_final, YTrain_tensors)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# LSTM 네트워크 구성 #
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # LSTM
        self.fc_1 = nn.Linear(hidden_size, 124)  # fully connected 1
        self.fc = nn.Linear(124, num_classes)  # fully connected last layer

#        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        # internal state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        # Propagate input through LSTM

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # First dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final output

        return out


lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers,
              XTrain_tensors_final.shape[1]).to(device)

loss_function = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01)


# 학습 #
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(dataloader):
        outputs = lstm1.forward(XTrain_tensors_final.to(device))  # forward pass
        optimizer.zero_grad()  # calculate the gradient. manually setting to 0

        # obtain the loss function
        loss = loss_function(outputs, YTrain_tensors.to(device))
        loss.backward()  # calculate the loss of the loss function
        optimizer.step()  # improve from loss. i.e backprop
#        scheduler.step(loss)

    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# 예측 #
df_x_ss = ss.transform(XTest)
df_y_mm = mm.transform(YTest)

df_x_ss = Variable(torch.Tensor(df_x_ss))  # converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))  # reshaping the dataset
df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))

train_predict = lstm1(df_x_ss.to(device))  # forward pass
data_predict = train_predict.data.detach().cpu().numpy()  # numpy conversion
data_y_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict)  # reverse transformation
data_y_plot = mm.inverse_transform(data_y_plot)


# 결과 확인 #
end = time.time()
sec = end - start
process_time = str(datetime.timedelta(seconds=sec)).split(".")
print(process_time[0])

plt.plot(data_y_plot, label='Actual Data')
plt.plot(data_predict, label='Predicted Data')
plt.legend()
plt.show()


# data 저장
file_out_raw = str(num_epochs) + '_' + str(learning_rate) + '_' + str(batch_size) \
               + '_' + str(hidden_size) + '_' + file_test
file_out = os.path.join(path_out, file_out_raw)
data = np.savetxt(file_out, data_predict, delimiter=',')



