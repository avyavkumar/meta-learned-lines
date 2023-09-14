import copy

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
        self.linear = nn.Linear(BERT_DIMS, classes)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def initialise_optimizer(self, optimizer):
        self.learning_rates_dict = copy.deepcopy(optimizer.learning_rates_dict)

    # def get_inner_loop_parameter_dict(self):
    #     """
    #     Returns a dictionary with the parameters to use for inner loop updates.
    #     :param params: A dictionary of the network's parameters.
    #     :return: A dictionary of the parameters to use for the inner loop optimization process.
    #     """
    #     return {
    #         name: param.to(device=self.device) for name, param in self.named_parameters() if param.requires_grad
    #     }

    # def assign_dict_to_params(self, weights_dict):
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             param = nn.Parameter(weights_dict[name])
    #             param.requires_grad_()

    def update_weights(self, grads_dict, num_step):
        for name, param in self.metaLearner.named_parameters():
            if param.requires_grad:
                if "LayerNorm" in name:
                    # don't update, just pass on
                    pass
                else:
                    print("old ", param)
                    param = param - self.learning_rates_dict["metaLearner-" + name.replace(".", "-")][num_step] * grads_dict["metaLearner." + name]
                    print(param - self.learning_rates_dict["metaLearner-" + name.replace(".", "-")][num_step] * grads_dict["metaLearner." + name])
        for name, param in self.linear.named_parameters():
            if param.requires_grad:
                param = nn.Parameter(param - self.learning_rates_dict["linear-" + name.replace(".", "-")][num_step] * grads_dict["linear." + name])

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
