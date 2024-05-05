import warnings

import torch
from torch import nn as nn

class BidirectionalRNN(nn.Module):
    
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BidirectionalRNN, self).__init__()

        self.rnn = nn.RNN(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent

class BidirectionalGRU(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent
    
    def load_state_dict(self, state_dict, strict=True):
        self.rnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.rnn.state_dict(), filename)

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, dropout=0, num_layers=1):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True,
                           dropout=dropout, num_layers=num_layers)
        #self.embedding = nn.Linear(nHidden * 2, nOut)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        #b, T, h = recurrent.size()
        #t_rec = recurrent.contiguous().view(b * T, h)

        #output = self.embedding(t_rec)  # [T * b, nOut]
        #output = output.view(b, T, -1)
        return recurrent
