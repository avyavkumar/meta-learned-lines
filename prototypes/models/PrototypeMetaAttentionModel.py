import numpy as np
import torch
from torch import nn

from attention.BahdanauAttention import BahdanauAttention
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from utils.Constants import BERT_DIMS

from prototypes.PrototypeModel import PrototypeModel
from utils.ModelUtils import DEVICE


class PrototypeMetaAttentionModel(nn.Module, PrototypeModel):
    def __init__(self):
        super(PrototypeMetaAttentionModel, self).__init__()
        self.metaLearner = PrototypeMetaModel()
        self.attention = BahdanauAttention(BERT_DIMS)
        self.linear2_1 = nn.Linear(BERT_DIMS, 2)
        torch.nn.init.xavier_uniform_(self.linear2_1.weight, gain=0.6)
        self.linear2_2 = nn.Linear(BERT_DIMS, 2)
        torch.nn.init.xavier_uniform_(self.linear2_2.weight, gain=1.4)
        self.linear3_1 = nn.Linear(BERT_DIMS, 3)
        torch.nn.init.xavier_uniform_(self.linear3_1.weight, gain=0.6)
        self.linear3_2 = nn.Linear(BERT_DIMS, 3)
        torch.nn.init.xavier_uniform_(self.linear3_2.weight, gain=1.4)

    def forward(self, inputs, labels, prototypeLabel_1, prototypeLabel_2):
        classes = len(set(labels))
        encodings = self.metaLearner(inputs)
        prototype_1 = torch.mean(torch.stack([encodings[i] for i in range(len(inputs)) if labels[i] == prototypeLabel_1]), dim=0)
        prototype_2 = torch.mean(torch.stack([encodings[i] for i in range(len(inputs)) if labels[i] == prototypeLabel_2]), dim=0)
        _, weights = self.attention(encodings.unsqueeze(1), torch.stack([prototype_1.to(DEVICE), prototype_2.to(DEVICE)], dim=0))
        weights = weights.squeeze(1)
        if classes == 2:
            outputs = weights[:, 0].unsqueeze(1) * self.linear2_1(encodings) + weights[:, 1].unsqueeze(1) * self.linear2_2(encodings)
        else:
            outputs = weights[:, 0].unsqueeze(1) * self.linear3_1(encodings) + weights[:, 1].unsqueeze(1) * self.linear3_2(encodings)
        return outputs

    def forward_test(self, supportSet, supportLabels, inputs, classes, prototypeLabel_1, prototypeLabel_2):
        encodings = self.metaLearner(inputs)
        prototype_1 = torch.mean(self.metaLearner([supportSet[x] for x in range(len(supportSet)) if supportLabels[x] == prototypeLabel_1]), dim=0)
        prototype_2 = torch.mean(self.metaLearner([supportSet[x] for x in range(len(supportSet)) if supportLabels[x] == prototypeLabel_2]), dim=0)
        _, weights = self.attention(encodings.unsqueeze(1), torch.stack([prototype_1, prototype_2], dim=0))
        weights = weights.squeeze(1)
        if classes == 2:
            outputs = weights[:, 0].unsqueeze(1) * self.linear2_1(encodings) + weights[:, 1].unsqueeze(1) * self.linear2_2(encodings)
        else:
            outputs = weights[:, 0].unsqueeze(1) * self.linear3_1(encodings) + weights[:, 1].unsqueeze(1) * self.linear3_2(encodings)
        return outputs

    def initialiseOuterLayers(self):
        torch.nn.init.xavier_uniform_(self.linear2_1.weight, gain=0.6)
        torch.nn.init.xavier_uniform_(self.linear2_2.weight, gain=1.4)
