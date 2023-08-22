import torch
from torch import nn

from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from utils.Constants import HIDDEN_MODEL_SIZE, BERT_DIMS, META_OUTPUT_SIZE

from prototypes.PrototypeModel import PrototypeModel
from utils.ModelUtils import DEVICE


class PrototypeMetaLinearModel(nn.Module, PrototypeModel):

    def __init__(self, metaLearner, classes):
        super(PrototypeMetaLinearModel, self).__init__()
        self.metaLearner = metaLearner
        self.linear = nn.Linear(META_OUTPUT_SIZE, classes)
        # torch.nn.init.xavier_uniform_(self.linear.weight)
        self.initialWeight = self.linear.weight.data
        self.initialBias = self.linear.bias.data

    def forward(self, inputs):
        metaOutputLayer = self.metaLearner(inputs)
        outputLayer = self.linear(metaOutputLayer)
        return outputLayer

    def getPrototypicalEmbedding(self, inputs):
        metaOutputLayer = self.metaLearner(inputs)
        return metaOutputLayer

    def getEncoder(self):
        return "None"

    def addToComputationGraph(self):
        self.linear.weight.data = (self.linear.weight.data - self.initialWeight.to(DEVICE)).detach() + self.initialWeight.requires_grad_().to(DEVICE)
        self.linear.bias.data = (self.linear.bias.data - self.initialBias.to(DEVICE)).detach() + self.initialBias.requires_grad_().to(DEVICE)

    def scaleGradients(self, scalingFactor):
        self.metaLearner.scaleGradients(scalingFactor)
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor

    def setParamsOfLinearLayer(self, weights, bias):
        self.linear.weight.data = weights
        self.linear.bias.data = bias
        self.linear.weight.requires_grad_()
        self.linear.bias.requires_grad_()
