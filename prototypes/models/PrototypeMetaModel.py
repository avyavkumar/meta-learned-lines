import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from utils.ModelUtils import get_prototypes

from prototypes.PrototypeModel import PrototypeModel

PROTONET = "ProtoNet"
PROTOFOMAML = "ProtoFOMAML"
FOMAML = "FOMAML"
HIDDEN_MODEL_SIZE = 256
BERT_DIMS = 768


class PrototypeMetaModel(nn.Module, PrototypeModel):

    # initialise BERT and output hidden layer
    # the classifier size is given as a parameter
    # for ProtoFOMAML - the weights need to be adjusted
    # outer loop and inner loop learning rates need to be an input

    def __init__(self, metaLearner, classes):
        super(PrototypeMetaModel, self).__init__()
        self.metaLearner = metaLearner
        self.classes = classes
        self.isOutputLayerInitialised = False if self.metaLearner == PROTOFOMAML else True
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.hidden = nn.Linear(BERT_DIMS, HIDDEN_MODEL_SIZE)
        self.output = nn.Linear(HIDDEN_MODEL_SIZE, classes)
        self.relu = nn.ReLU()

    def forward(self, inputs, labels):
        if self.isOutputLayerInitialised is False:
            self.initialiseOutputLayer(inputs, labels)
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt")
        outputs = self.bert(**tokenized_inputs)
        encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
        hidden_output = self.hidden(encoding)
        dropout = self.dropout(hidden_output)
        output = self.output(dropout)
        return self.relu(output)

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.hidden.weight.grad = self.hidden_1.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor
        self.hidden.bias.grad = self.hidden_1.bias.grad * scalingFactor

    def initialiseOutputLayer(self, inputs, labels):
        if self.metaLearner == PROTOFOMAML:
            # adjust weights
            prototypes, unique_labels = get_prototypes(inputs, labels)
            init_weight = 2 * prototypes
            init_bias = -torch.norm(prototypes, dim=1) ** 2
            output_weight = init_weight.detach().requires_grad_()
            output_bias = init_bias.detach().requires_grad_()
            self.isOutputLayerInitialised = True
