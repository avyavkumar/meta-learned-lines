import torch
import torch.nn as nn
import torch.nn.functional as F


# source - https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        # self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(self.tanh(self.Ua(query + keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        # weights = F.softmax(scores, dim=-1)
        context = torch.matmul(scores, keys)

        return context, scores
