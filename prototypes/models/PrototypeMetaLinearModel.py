import torch
from torch import nn
from utils.Constants import HIDDEN_MODEL_SIZE

from prototypes.PrototypeModel import PrototypeModel


class PrototypeMetaLinearModel(nn.Module, PrototypeModel):

    def __init__(self, metaLearner, classes):
        super(PrototypeMetaLinearModel, self).__init__()
        self.metaLearner = metaLearner
        # Apply a seed to reproduce exact results
        # torch.manual_seed(42)
        self.linear = nn.Linear(HIDDEN_MODEL_SIZE, classes)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, inputs):
        metaOutputLayer = self.metaLearner(inputs)
        outputLayer = self.linear(metaOutputLayer)
        return outputLayer

    def getEncoder(self):
        return "None"

    def scaleGradients(self, scalingFactor):
        # self.metaLearner.hidden.weight.data = self.metaLearner.hidden.weight.data * scalingFactor
        # self.metaLearner.hidden.bias.data = self.metaLearner.hidden.bias.data * scalingFactor
        # print(self.metaLearner.hidden.weight.data)
        # print(self.metaLearner.hidden.bias.data)
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor
