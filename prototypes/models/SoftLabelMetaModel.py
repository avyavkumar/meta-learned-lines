from torch import nn
from transformers import BertModel, BertTokenizer

from utils import ModelUtils
from utils.Constants import BERT_DIMS, HIDDEN_MODEL_SIZE, META_OUTPUT_SIZE

from prototypes.PrototypeModel import PrototypeModel


class SoftLabelMetaModel(nn.Module, PrototypeModel):

    def __init__(self):
        super(SoftLabelMetaModel, self).__init__()
        self.hidden = nn.Linear(BERT_DIMS, META_OUTPUT_SIZE)

    def forward(self, inputs):
        hidden_output = self.hidden(inputs)
        return hidden_output

    def getEncoder(self):
        return "2NN"

    def scaleGradients(self, scalingFactor):
        self.hidden.weight.grad = self.hidden.weight.grad * scalingFactor
        self.hidden.bias.grad = self.hidden.bias.grad * scalingFactor
