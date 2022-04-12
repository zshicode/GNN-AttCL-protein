import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class ConvLstm(nn.Module):
    ''' Define the model using CNN and LSTM '''
    def __init__(self, learning_rate):
        super(ConvLstm, self).__init__()
        self.kernel_size_1 = 2
        self.pool_size_1 = 2
        self.convPool1 = nn.Sequential(
            nn.Conv1d(50, 30, self.kernel_size_1), nn.ReLU(True), nn.MaxPool1d(self.pool_size_1, self.pool_size_1)
        )
        self.drop1 = nn.Dropout(0.2)

        self.kernel_size_2 = 1
        self.pool_size_2 = 2
        self.convPool2 = nn.Sequential(
            nn.Conv1d(30, 30, self.kernel_size_2), nn.ReLU(True), nn.MaxPool1d(self.pool_size_2, self.pool_size_2)
        )
        self.drop2 = nn.Dropout(0.2)

        # Bi-LSTM layer definition
        self.BiLstm = nn.LSTM(
            input_size = 30,    # input size
            hidden_size = 30,   # hidden size of lstm
            num_layers = 1,     # num of layers
            batch_first = True,
            bidirectional = True  # two directions of LSTM
        )
        self.lstmdrop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * 30, 32)  # fc layer1 160 * 2
        self.fcdrop = nn.Dropout(0.5)
        self.wq = nn.Linear(32,32)
        self.wk = nn.Linear(32,32)
        self.wv = nn.Linear(32,32)
        self.fc2 = nn.Linear(32, 6)       # fc layer2 32 * 6 (six classes)

        self.criterion = nn.MultiLabelSoftMarginLoss()  # set criterion for training
        self.optimier = optim.Adam(self.parameters(), lr=learning_rate)   # set optimizer for optimization


    def forward(self, x, seq_lengths):
        # permute tensor to convolute it in the last dimension
        x = x.permute(0, 2, 1)
        # The three layers with conv, relu and maxpool
        x = self.convPool1(x)
        x = self.drop1(x)

        x = self.convPool2(x)
        x = self.drop2(x)
        
        x = x.permute(0, 2 ,1)  # batch * length * feature
        _, (x, _) = self.BiLstm(x)  # x: hidden variables, layers * batch * feature
        x = x.permute(1, 0 ,2)
        x = self.lstmdrop(x)
        # flatten the tensor into fc layer
        # use .reshape() rather than .size() because .reshape() will automatically make the tensor contiguous
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fcdrop(x)
        # q = self.wq(x)
        # k = self.wk(x)
        # v = self.wv(x)
        # x = torch.mm(torch.softmax(torch.mm(q,k.t())/5.65,dim=-1),v)#sqrt(32)=5.65
        x = self.fc2(x)
        return x
