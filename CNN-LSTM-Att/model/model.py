import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM_ATT(nn.Module):
    def __init__(self, window=3, dim=300, lstm_units=128, num_layers=1, out_conv_filters=1000):
        super(CNN_LSTM_ATT, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=dim, out_channels=out_conv_filters, kernel_size=window,padding=2)
        self.act1 = nn.ReLU()
        # self.maxPool = nn.MaxPool1d(kernel_size=5)
        self.maxPool = nn.MaxPool1d(kernel_size=20)
        self.avgPool = nn.AvgPool1d(kernel_size=15)
        self.lstm = nn.LSTM(input_size=out_conv_filters, hidden_size=lstm_units, batch_first=True, num_layers=2,
                            dropout=0.3,bidirectional=False)
        self.attn = nn.Linear(lstm_units, lstm_units)
        self.act2 = nn.Softmax()
        self.cls = nn.Linear(lstm_units, 1)


    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        # x = self.maxPool(x)  # bs, lstm_units, 1
        x1 = self.maxPool(x)  # bs, lstm_units, 1
        x2 = self.avgPool(x)
        x = torch.cat((x1,x2),dim=2)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        attn = self.attn(x)  # bs, 2*lstm_units
        attn = self.act2(attn)
        x = x * attn
        x = self.cls(x)
        return x
