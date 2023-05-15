from torch import nn

from prototypes.PrototypeModel import PrototypeModel

PROTONET = "ProtoNet"
PROTOFOMAML = "ProtoFOMAML"
FOMAML = "FOMAML"
HIDDEN_MODEL_SIZE = 256


class PrototypeMetaModel(nn.Module, PrototypeModel):

    # initialise BERT and output hidden layer
    # the classifier size is given as a parameter
    # for ProtoFOMAML - the weights need to be adjusted
    # outer loop and inner loop learning rates need to be an input

    def __init__(self, metaLearner, classes):
        super(PrototypeMetaModel, self).__init__()
        self.metaLearner = metaLearner
        self.classes = classes
        if self.metaLearner == PROTOFOMAML:
            self.initialiseLinearLayer()

    def forward(self, inputs):
        pass

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.hidden.weight.grad = self.hidden_1.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor
        self.hidden.bias.grad = self.hidden_1.bias.grad * scalingFactor

    def initialiseLinearLayer(self):
        if self.metaLearner == PROTOFOMAML:
            # adjust weights
            pass
