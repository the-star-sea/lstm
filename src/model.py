# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class TextGenerator(nn.ModuleList):
    def __init__(self, args, vocab_size,predict=False):
        super(TextGenerator, self).__init__()

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_dim
        self.attention_size=int(args.hidden_dim/2)
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = args.window
        self.layer_size=args.layer_size
        self.embed_size = args.embed_size
        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        if predict:
            self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.layer_size,
                                batch_first=False, dropout=0, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.layer_size,
                            batch_first=False,dropout=0.25, bidirectional=True)
        self.layer_size =self.layer_size*2

        self.fc = nn.Linear(self.hidden_size,self.num_classes)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True)
        )




    def forward(self, x, hidden=None,predict=False):


        out = self.embedding(x)
        out = out.view(self.sequence_len, x.size(0), -1)

        if hidden is None:
            if predict:
                h_0 = x.data.new(self.layer_size, 1, self.hidden_size).fill_(0).float().to('cuda')
                c_0 = x.data.new(self.layer_size, 1, self.hidden_size).fill_(0).float().to('cuda')
            else:
                h_0 = x.data.new(self.layer_size, self.batch_size, self.hidden_size).fill_(0).float().to('cuda')
                c_0 = x.data.new(self.layer_size, self.batch_size, self.hidden_size).fill_(0).float().to('cuda')
        else:
            h_0, c_0 = hidden

        out, (h_n, c_n) = self.lstm(out, (h_0, c_0))#hidden 是h,和c 这两个隐状态
        hidden=(h_n, c_n)
        (forward_out, backward_out) = torch.chunk(out, 2, dim=2)
        out = forward_out + backward_out  # [seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  # [batch, seq_len, hidden_size]

        # 为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  # [batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1)  # [batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  # [batch, hidden_size]

        attention_w = self.attention_weights_layer(h_n)  # [batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1)  # [batch, 1, hidden_size]

        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  # [batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  # [batch, 1, seq_len],权重归一化

        x = torch.bmm(softmax_w, out)  # [batch, 1, hidden_size]
        x = x.squeeze(dim=1)
        out = self.fc(x)
        return out,hidden
