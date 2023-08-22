from torch import nn
from transformers import BertModel, BertTokenizer

from utils import ModelUtils
from utils.Constants import BERT_DIMS, HIDDEN_MODEL_SIZE, META_OUTPUT_SIZE

from prototypes.PrototypeModel import PrototypeModel


class SoftLabelPrototypeMetaModel(nn.Module, PrototypeModel):

    def __init__(self, prototypeEmbeddingMetaModel, softLabelMetaModel):
        super(SoftLabelPrototypeMetaModel, self).__init__()
        self.prototypeEmbeddingMetaModel = prototypeEmbeddingMetaModel
        self.softLabelMetaModel = softLabelMetaModel

    def forward(self, inputs):
        location = self.prototypeEmbeddingMetaModel(inputs)
        softLabels = self.softLabelMetaModel(location)
        return softLabels

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        self.prototypeEmbeddingMetaModel.scaleGradients(scalingFactor)
        self.softLabelMetaModel.scaleGradients(scalingFactor)
