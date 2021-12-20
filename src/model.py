# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerator(nn.ModuleList):
    def __init__(self, args, vocab_size):
        super(TextGenerator, self).__init__()

        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = args.window

        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=3,
                            batch_first=False,dropout=0.25, bidirectional=True)
        self.fc1 = nn.Linear(2*self.hidden_dim,2048)
        self.fc2 = nn.Linear(2048,4096)
        self.fc3 = nn.Linear(4096,self.num_classes)


    def forward(self, x, hidden=None):


        out = self.embedding(x)
        out = out.view(self.sequence_len, x.size(0), -1)
        if hidden is None:
            h_0 = x.data.new(2*3, self.batch_size, self.hidden_dim).fill_(0).float().to('cuda')
            c_0 = x.data.new(2*3, self.batch_size, self.hidden_dim).fill_(0).float().to('cuda')
        else:
            h_0, c_0 = hidden
        # print(out.shape)
        output, hidden = self.lstm(out, (h_0, c_0))#hidden 是h,和c 这两个隐状态
        # output=output.view(self.sequence_len * self.batch_size, -1)
        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output)
        # output = output.reshape(self.batch_size* self.sequence_len, -1)
        return output,hidden
