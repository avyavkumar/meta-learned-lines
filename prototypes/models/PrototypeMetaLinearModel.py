import copy

import torch
from torch import nn
import numpy as np

from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from utils.Constants import HIDDEN_MODEL_SIZE, BERT_DIMS, META_OUTPUT_SIZE

from prototypes.PrototypeModel import PrototypeModel
from utils.ModelUtils import DEVICE, get_prototypes


class PrototypeMetaLinearModel(nn.Module, PrototypeModel):

    def __init__(self, metaLearner, classes):
        super(PrototypeMetaLinearModel, self).__init__()
        self.metaLearner = metaLearner
        self.linear = nn.Linear(BERT_DIMS, classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        metaOutputLayer = self.metaLearner(inputs)
        dropout = self.dropout(metaOutputLayer)
        outputLayer = self.linear(dropout)
        return outputLayer

    def getPrototypicalEmbedding(self, inputs):
        metaOutputLayer = self.metaLearner(inputs)
        return metaOutputLayer

    def getEncoder(self):
        return "None"

    def scaleGradients(self, scalingFactor):
        self.metaLearner.scaleGradients(scalingFactor)
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor

    def setParamsOfLinearLayer(self, prototypes, uniqueLabels):
        with torch.no_grad():
            self.linear[len(uniqueLabels)].weight.data.copy_(2 * prototypes)
            self.linear[len(uniqueLabels)].bias.data.copy_(-torch.norm(prototypes, dim=1) ** 2)
            # add some noise
            for param in self.linear.parameters():
                param.add_(torch.randn(param.size()) * 0.1)

class PrototypeMetaLinearUnifiedModel(nn.Module, PrototypeModel):

    def __init__(self, metaLearner, classes, protoFOMAML=False):
        super(PrototypeMetaLinearUnifiedModel, self).__init__()
        self.metaLearner = metaLearner
        self.protoFOMAML = protoFOMAML
        self.initialised = False
        self.linear_1 = nn.Linear(BERT_DIMS, classes)
        self.linear_2 = nn.Linear(BERT_DIMS, classes)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.initialiseOuterLayers()

    def forward(self, inputs, labels, prototypeLabel_1, prototypeLabel_2):
        if self.protoFOMAML and not self.initialised:
            with torch.no_grad():
                prototypes, _ = get_prototypes(self.metaLearner(inputs), torch.LongTensor(labels))
                self.linear_1.weight.data.copy_(2 * prototypes)
                self.linear_1.bias.data.copy_(-torch.norm(prototypes, dim=1) ** 2 / BERT_DIMS)
                self.linear_2.weight.data.copy_(2 * prototypes)
                self.linear_2.bias.data.copy_(-torch.norm(prototypes, dim=1) ** 2 / BERT_DIMS)
            self.initialised = True
        encodings = self.metaLearner(inputs)
        prototype_1 = torch.mean(torch.stack([encodings[i] for i in range(len(inputs)) if labels[i] == prototypeLabel_1]), dim=0)
        prototype_2 = torch.mean(torch.stack([encodings[i] for i in range(len(inputs)) if labels[i] == prototypeLabel_2]), dim=0)
        distances_1 = []
        distances_2 = []
        for i in range(encodings.shape[0]):
            distances_1.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_1.detach().cpu().numpy()))
            distances_2.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_2.detach().cpu().numpy()))
        distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1).to(DEVICE)
        distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1).to(DEVICE)
        outputs = self.linear_1(encodings) / distances_1 + self.linear_2(encodings) / distances_2
        return outputs, distances_1, distances_2

    def forward_test(self, supportSet, supportLabels, inputs, prototypeLabel_1, prototypeLabel_2):
        encodings = self.metaLearner(inputs)
        prototype_1 = torch.mean(self.metaLearner([supportSet[x] for x in range(len(supportSet)) if supportLabels[x] == prototypeLabel_1]), dim=0)
        prototype_2 = torch.mean(self.metaLearner([supportSet[x] for x in range(len(supportSet)) if supportLabels[x] == prototypeLabel_2]), dim=0)
        distances_1 = []
        distances_2 = []
        for i in range(encodings.shape[0]):
            distances_1.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_1.detach().cpu().numpy()))
            distances_2.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_2.detach().cpu().numpy()))
        distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1).to(DEVICE)
        distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1).to(DEVICE)
        outputs = self.linear_1(encodings) / distances_1 + self.linear_2(encodings) / distances_2
        return outputs

    def getPrototypicalEmbedding(self, inputs):
        metaOutputLayer = self.metaLearner(inputs)
        return metaOutputLayer

    def getEncoder(self):
        return "None"

    def scaleModelGradients(self, scalingFactor_1, scalingFactor_2):
        self.linear_1.weight.grad = self.linear_1.weight.grad * scalingFactor_1
        self.linear_1.bias.grad = self.linear_1.bias.grad * scalingFactor_1
        self.linear_2.weight.grad = self.linear_2.weight.grad * scalingFactor_2
        self.linear_2.bias.grad = self.linear_2.bias.grad * scalingFactor_2

    def initialiseOuterLayers(self, protoFOMAML=False):
        torch.nn.init.xavier_uniform_(self.linear_2.weight, gain=1.4)
