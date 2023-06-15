import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from utils.ModelUtils import get_prototypes
from utils.Constants import BERT_DIMS, HIDDEN_MODEL_SIZE, PROTOFOMAML

from prototypes.PrototypeModel import PrototypeModel


class PrototypeMetaModel(nn.Module, PrototypeModel):

    # initialise BERT and output hidden layer
    # the classifier size is given as a parameter
    # for ProtoFOMAML - the weights need to be adjusted
    # outer loop and inner loop learning rates need to be an input

    def __init__(self):
        super(PrototypeMetaModel, self).__init__()
        # self.isOutputLayerInitialised = False if self.metaLearner == PROTOFOMAML else True
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hidden = nn.Linear(BERT_DIMS, HIDDEN_MODEL_SIZE)

    def forward(self, inputs, labels):
        if self.isOutputLayerInitialised is False:
            self.initialiseOutputLayer(inputs, labels)
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt")
        outputs = self.bert(**tokenized_inputs)
        encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
        hidden_output = self.hidden(encoding)
        return self.relu(hidden_output)

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        self.linear.weight.grad = self.linear.weight.grad * scalingFactor
        self.hidden.weight.grad = self.hidden_1.weight.grad * scalingFactor
        self.linear.bias.grad = self.linear.bias.grad * scalingFactor
        self.hidden.bias.grad = self.hidden_1.bias.grad * scalingFactor

    # def initialiseOutputLayer(self, inputs, labels):
    #     if self.metaLearner == PROTOFOMAML:
    #         # adjust weights
    #         prototypes, unique_labels = get_prototypes(inputs, labels)
    #         self.output = nn.Linear(HIDDEN_MODEL_SIZE, len(set(labels)))
    #         init_weight = 2 * prototypes
    #         init_bias = -torch.norm(prototypes, dim=1) ** 2
    #         output_weight = init_weight.detach().requires_grad_()
    #         output_bias = init_bias.detach().requires_grad_()
    #         self.output.weight = output_weight
    #         self.output.bias = output_bias
    #         self.isOutputLayerInitialised = True
