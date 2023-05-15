import torch.nn.init
from torch import nn
from prototypes.PrototypeModel import PrototypeModel

CLASSIFIER_MODEL_2NN = "classifier_2NN"
CLASSIFIER_MODEL_3NN = "classifier_3NN"
CLASSIFIER_MODEL_4NN = "classifier_4NN"
HIDDEN_MODEL_SIZE = 1024


# TODO check if softmax can be removed
# TODO experiment with the architecture of the classifier
class PrototypeClassifierModel4NN(nn.Module, PrototypeModel):

    def __init__(self, input_dims, classes):
        super(PrototypeClassifierModel4NN, self).__init__()
        self.classes = classes
        self.hidden_1 = nn.Linear(input_dims, HIDDEN_MODEL_SIZE)
        self.hidden_2 = nn.Linear(HIDDEN_MODEL_SIZE, HIDDEN_MODEL_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(HIDDEN_MODEL_SIZE, classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        hidden_layer_1 = self.hidden_1(inputs)
        hidden_layer_1 = self.relu(hidden_layer_1)
        dropout_1 = self.dropout(hidden_layer_1)
        hidden_layer_2 = self.hidden_2(dropout_1)
        hidden_layer_2 = self.relu(hidden_layer_2)
        dropout_2 = self.dropout(hidden_layer_2)
        linear_layer = self.linear(dropout_2)
        # output_layer = self.softmax(linear_layer)
        output_layer = linear_layer
        return output_layer

    def getEncoder(self):
        return "None"

    def scaleGradients(self, scalingFactor):
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.hidden_1.weight.grad = self.hidden_1.weight.grad * scalingFactor
        self.hidden_2.weight.grad = self.hidden_2.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor
        self.hidden_1.bias.grad = self.hidden_1.bias.grad * scalingFactor
        self.hidden_2.bias.grad = self.hidden_2.bias.grad * scalingFactor


class PrototypeClassifierModel3NN(nn.Module, PrototypeModel):

    def __init__(self, input_dims, classes):
        super(PrototypeClassifierModel3NN, self).__init__()
        self.classes = classes
        self.hidden = nn.Linear(input_dims, HIDDEN_MODEL_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(HIDDEN_MODEL_SIZE, classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        hidden_layer = self.hidden(inputs)
        hidden_layer = self.relu(hidden_layer)
        dropout = self.dropout(hidden_layer)
        linear_layer = self.linear(dropout)
        output_layer = linear_layer
        return output_layer

    def getEncoder(self):
        return "None"

    def scaleGradients(self, scalingFactor):
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.hidden.weight.grad = self.hidden_1.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor
        self.hidden.bias.grad = self.hidden_1.bias.grad * scalingFactor


class PrototypeClassifierModel2NN(nn.Module, PrototypeModel):

    def __init__(self, input_dims, classes):
        super(PrototypeClassifierModel2NN, self).__init__()
        self.classes = classes
        # Apply a seed to reproduce exact results
        # torch.manual_seed(42)
        self.linear = nn.Linear(input_dims, classes)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, inputs):
        output_layer = self.linear(inputs)
        return output_layer

    def getEncoder(self):
        return "None"

    def scaleGradients(self, scalingFactor):
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor
