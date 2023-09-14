import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(GradientDescentLearningRule, self).__init__()
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = torch.ones(1) * learning_rate
        self.learning_rate.to(device)

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.9):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        return {
            key: names_weights_dict[key]
                 - self.learning_rate * names_grads_wrt_params_dict[key]
            for key in names_weights_dict.keys()
        }


class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, total_num_inner_loop_steps, init_learning_rate=1e-5, init_class_head_lr_multiplier=10):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()
        assert init_learning_rate > 0.0, "learning_rate should be positive."

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_class_head_lr_multiplier = init_class_head_lr_multiplier
        # self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps

    def initialise(self, named_parameters):
        names_weights_dict = self.convert_named_parameters_to_dict(named_parameters)
        self.learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            multiplier = 1
            if "linear" in key:
                multiplier = self.init_class_head_lr_multiplier
            self.learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps) * self.init_learning_rate * multiplier,
                requires_grad=param.requires_grad,
            )

    def convert_named_parameters_to_dict(self, named_parameters):
        parameters_dict = {}
        for name, param in named_parameters:
            if param.requires_grad:
                parameters_dict[name] = param
        return parameters_dict

    def reset(self):

        # for key, param in self.names_learning_rates_dict.items():
        #     param.fill_(self.init_learning_rate)
        pass

    def update_params(self, weights_dict, grads_dict, num_step):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        # names_weights_dict = self.convert_named_parameters_to_dict(named_parameters)
        # names_grads_wrt_params_dict = self.convert_named_parameter_grads_to_dict(named_parameters)
        updated_names_weights_dict = dict()
        # device = names_weights_dict[next(iter(names_weights_dict.keys()))].device
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if "LayerNorm" not in name:
        #             param = param - self.learning_rates_dict[name.replace(".", "-")][num_step] * grads_dict[name]
        #             # if "11.output.dense.weight" in name:
        #             #     print(name, param.shape, grads_dict[name])
        #             # with torch.no_grad():
        #             #     param.copy_(updated_param)
        #             param.requires_grad_()
        #

        for key in grads_dict.keys():
            if "LayerNorm" in key:
                # don't update, just pass on
                updated_names_weights_dict[key] = weights_dict[key]
            else:
                updated_names_weights_dict[key] = weights_dict[key] - self.learning_rates_dict[key.replace(".", "-")][num_step] * grads_dict[key]

        return updated_names_weights_dict
