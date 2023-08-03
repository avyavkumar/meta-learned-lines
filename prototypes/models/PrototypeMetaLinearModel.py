import torch
from torch import nn

from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from utils.Constants import HIDDEN_MODEL_SIZE, BERT_DIMS

from prototypes.PrototypeModel import PrototypeModel


class PrototypeMetaLinearModel(nn.Module, PrototypeModel):

    def __init__(self, metaLearner: PrototypeMetaModel, classes):
        super(PrototypeMetaLinearModel, self).__init__()
        self.metaLearner = metaLearner
        self.linear = nn.Linear(BERT_DIMS, classes)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, inputs):
        metaOutputLayer = self.metaLearner(inputs)
        outputLayer = self.linear(metaOutputLayer)
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

    def setParamsOfLinearLayer(self, weights, bias):
        self.linear.weight.data = weights
        self.linear.bias.data = bias
        self.linear.weight.requires_grad_()
        self.linear.bias.requires_grad_()
