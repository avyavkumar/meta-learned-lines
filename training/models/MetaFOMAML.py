import random
from copy import deepcopy

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchviz import make_dot
from transformers import get_constant_schedule_with_warmup

from higher.optim import DifferentiableSGD
from lines.Line import Line
from lines.LineGenerator import LineGenerator
from lines.lo_shot_utils import dist_to_line_multiD
from optimisers.inner_loop_optimisers import LSLRGradientDescentLearningRule
from prototypes.models.PrototypeMetaLinearModel import PrototypeMetaLinearModel
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils import ModelUtils
from utils.Constants import PROTOTYPE_META_MODEL
from utils.ModelUtils import DEVICE, CPU_DEVICE


# TODO check if data is correct and training is going as expected
# TODO log metrics of individual validation episodes
# TODO load model from checkpoint and check if everything works as expected
# TODO write code for distributed hyper param checking

class MetaFOMAML(L.LightningModule):

    def __init__(self, outerLR, innerLR, steps, batchSize, warmupSteps):
        super().__init__()
        self.predictions = []
        self.actualLabels = []
        self.losses = []
        self.save_hyperparameters()
        self.val_episode = 0
        self.automatic_optimization = False
        self.metaLearner = PrototypeMetaModel()
        self.innerLoopOptimisersInitialised = False
        self.optimizer_1 = LSLRGradientDescentLearningRule(DEVICE, self.hparams.steps, init_learning_rate=innerLR)
        self.optimizer_2 = LSLRGradientDescentLearningRule(DEVICE, self.hparams.steps, init_learning_rate=innerLR)
        self.optimizer_1.initialise(PrototypeMetaLinearModel(self.metaLearner, 3).named_parameters())
        self.optimizer_2.initialise(PrototypeMetaLinearModel(self.metaLearner, 3).named_parameters())
        self.optimizer_1.to(DEVICE)
        self.optimizer_2.to(DEVICE)
        torch.set_printoptions(threshold=100)

        optim = DifferentiableSGD(self.metaLearner.parameters())
        optim.param_groups()
        print("Training MetaFOMAML with parameters", self.hparams)

    def filterEncodingsByLabels(self, labels, training_data, training_labels):
        filteredTrainingData = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingData.append(training_data[i].detach().cpu().numpy())
                filteredTrainingLabels.append(training_labels[i])
        return torch.Tensor(np.array(filteredTrainingData)), np.array(filteredTrainingLabels)

    def configure_optimizers(self):
        optimiser = AdamW([{
            "params": self.metaLearner.parameters(),
            "lr": self.hparams.outerLR
        }, {
            "params": self.optimizer_1.parameters(),
            "lr": 6e-1
        }, {
            "params": self.optimizer_2.parameters(),
            "lr": 6e-1
        }], lr=self.hparams.outerLR)
        return {"optimizer": optimiser, "lr_scheduler": {"scheduler": CosineAnnealingWarmRestarts(optimizer=optimiser, T_0=1500, verbose=True)}}

    def updateGradients(self, losses, grads_1, grads_2, distances_1, distances_2):
        losses_1 = losses.clone().detach().cpu()
        losses_2 = losses.clone().detach().cpu()
        losses_1 = distances_2.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_1
        losses_2 = distances_1.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_2
        loss_ratio_1 = losses_1.sum() / (losses_1.sum() + losses_2.sum())
        loss_ratio_2 = losses_2.sum() / (losses_1.sum() + losses_2.sum())
        for grad in grads_1:
            grad *= loss_ratio_1
        for grad in grads_2:
            grad *= loss_ratio_2

    def getCriterion(self):
        return nn.CrossEntropyLoss(reduction='none')

    def computeLabelsAndDistances(self, inputs, encodings, model_1, model_2, location_1, location_2):
        output_1 = model_1(inputs).to(CPU_DEVICE)
        output_2 = model_2(inputs).to(CPU_DEVICE)
        # get distances from the prototypes for all inputs
        distances_1 = []
        distances_2 = []
        for i in range(encodings.shape[0]):
            distances_1.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - location_1.detach().cpu().numpy()))
            distances_2.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - location_2.detach().cpu().numpy()))
        distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1)
        distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1)
        # compute the weighted probability distribution
        outputs = output_1 / distances_1 + output_2 / distances_2
        # delete the outputs
        del output_1, output_2
        # return the final weighted probability distribution
        return outputs, distances_1, distances_2

    def filterSentencesByLabels(self, labels, training_data, training_labels):
        filteredTrainingSentences = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingSentences.append(training_data[i])
                filteredTrainingLabels.append(training_labels[i])
        return filteredTrainingSentences, np.array(filteredTrainingLabels)

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param.to(device=self.device) for name, param in params if param.requires_grad and "learning_rates_dict" not in name
        }

    def get_static_parameter_dict(self, state_dict, trainable_params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param.to(device=self.device) for name, param in state_dict.items() if name not in trainable_params
        }

    def runInnerLoopTrainingStep(self, line, model_1, model_2, filteredTrainingSentences, filteredTrainingEncodings, filteredTrainingLabels, step):
        trainingParams = {'batch_size': 32}
        trainingDataset = SentenceEncodingDataset(filteredTrainingSentences, filteredTrainingEncodings, filteredTrainingLabels)
        trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
        predictions = []
        correctLabels = []
        criterion = self.getCriterion()
        training_losses = []
        # model_1_static_parameters = {}
        # model_2_static_parameters = {}
        for i, data in enumerate(trainLoader, 0):
            # zero the parameter gradients
            model_1.zero_grad()
            model_2.zero_grad()
            # get the inputs; data is a list of [inputs, encodings, labels]
            inputs, encodings, labels = data
            outputs, distances_1, distances_2 = self.computeLabelsAndDistances(inputs, encodings, model_1, model_2, line.getFirstPrototype().getLocation(), line.getSecondPrototype().getLocation())
            # compute the loss
            losses = criterion(outputs, labels)
            model_1_trainable_parameters = self.get_inner_loop_parameter_dict(model_1.named_parameters())
            for name, param in model_1.named_parameters():
                print(name, param.requires_grad)
            # model_1_static_parameters = self.get_static_parameter_dict(model_1.state_dict(), model_1_trainable_parameters.keys())
            grads_1 = torch.autograd.grad(losses.sum(), model_1_trainable_parameters.values(), retain_graph=True, allow_unused=True)
            model_2_trainable_parameters = self.get_inner_loop_parameter_dict(model_2.named_parameters())
            # model_2_static_parameters = self.get_static_parameter_dict(model_2.state_dict(), model_2_trainable_parameters.keys())
            grads_2 = torch.autograd.grad(losses.sum(), model_2_trainable_parameters.values())
            self.updateGradients(losses, grads_1, grads_2, distances_1, distances_2)
            # for grad_1, grad_2 in zip(grads_1, grads_2):
            #     grad_1.detach()
            #     grad_2.detach()
            grads_1_dict = dict(zip(model_1_trainable_parameters.keys(), grads_1))
            grads_2_dict = dict(zip(model_2_trainable_parameters.keys(), grads_2))

            model_1.update_weights(grads_1_dict, step)
            model_2.update_weights(grads_2_dict, step)
            # updated_names_weights_dict_2 = self.optimizer_2.update_params(model_2_trainable_parameters, grads_2_dict, step)
            # model_1_static_parameters.update(updated_names_weights_dict_1)
            # model_2_static_parameters.update(updated_names_weights_dict_2)
            # model_1.load_state_dict(model_1_static_parameters)
            # model_2.load_state_dict(model_2_static_parameters)

            predictions_i = torch.argmax(outputs, dim=1).tolist()
            predictions.extend(predictions_i)
            correctLabels.extend(labels)
            training_losses.append(losses.sum().item())

            del outputs, distances_1, distances_2, grads_1, grads_1_dict, grads_2, grads_2_dict

        print("inner loop training loss is", sum(training_losses), "and accuracy is", accuracy_score(correctLabels, predictions))
        self.log("inner_loop_training_loss", sum(training_losses), batch_size=len(training_losses))
        self.log("inner_loop_training_accuracy", accuracy_score(correctLabels, predictions), batch_size=len(predictions))

    def trainInnerLoop(self, line: Line, supportSet, supportEncodings, supportLabels):
        # create models for inner loop updates
        innerLoopModel_1 = deepcopy(line.getFirstPrototype().getPrototypeModel())
        innerLoopModel_2 = deepcopy(line.getSecondPrototype().getPrototypeModel())
        # add these models to the GPU if there is a GPU
        innerLoopModel_1.to(ModelUtils.DEVICE)
        innerLoopModel_2.to(ModelUtils.DEVICE)
        innerLoopModel_1.initialise_optimizer(self.optimizer_1)
        innerLoopModel_2.initialise_optimizer(self.optimizer_2)
        # if not self.innerLoopOptimisersInitialised:
        #     self.optimizer_1.initialise(innerLoopModel_1.named_parameters())
        #     self.optimizer_2.initialise(innerLoopModel_2.named_parameters())
        #     self.innerLoopOptimisersInitialised = True
        # filter support encodings and labels to ensure that only line-specific data is used for training
        filteredTrainingSentences, filteredTrainingLabels_1 = self.filterSentencesByLabels(line.getLabels(), supportSet, supportLabels)
        filteredTrainingEncodings, filteredTrainingLabels_2 = self.filterEncodingsByLabels(line.getLabels(), supportEncodings, supportLabels)
        # sanity check to ensure that the filtered data is in the correct order
        assert torch.all(torch.eq(torch.tensor(filteredTrainingLabels_1, dtype=torch.int8), torch.tensor(filteredTrainingLabels_2, dtype=torch.int8))) == torch.tensor(True)

        for name, param in innerLoopModel_1.named_parameters():
            if param.requires_grad and "11.output.dense.weight" in name:
                print(name, param)

        # use SGD to carry out few-shot adaptation
        for step in range(self.hparams.steps):
            self.runInnerLoopTrainingStep(line, innerLoopModel_1, innerLoopModel_2, filteredTrainingSentences, filteredTrainingEncodings, filteredTrainingLabels_2, step)

        for name, param in innerLoopModel_1.named_parameters():
            if param.requires_grad and "11.output.dense.weight" in name:
                print(name, param)

        # delete everything to free the memory
        del filteredTrainingEncodings
        # put these models on the CPU again as part of the line
        innerLoopModel_1.to(CPU_DEVICE)
        innerLoopModel_2.to(CPU_DEVICE)
        line.getFirstPrototype().setPrototypeModel(innerLoopModel_1)
        line.getSecondPrototype().setPrototypeModel(innerLoopModel_2)

    def runOuterLoop(self, supportLines, supportEncodings, supportLabels, querySentences, queryEncodings, queryLabels, train=True):
        assignments = []
        outerLoopLoss = 0.0
        outerLoopPredictions = []
        outerLoopLabels = []
        for point in queryEncodings:
            dists = [dist_to_line_multiD(point.detach().cpu().numpy(), line.getFirstPrototype().getLocation().detach().cpu().numpy(), line.getSecondPrototype().getLocation().detach().cpu().numpy()) for line in supportLines]
            nearest = np.argmin(dists)
            assignments.append(nearest)
        for i in range(len(supportLines)):
            requiredQueryEncodings = np.array([queryEncodings[x].detach().cpu().numpy() for x in range((len(assignments))) if assignments[x] == i])
            requiredQuerySentences = [querySentences[x] for x in range((len(assignments))) if assignments[x] == i]
            requiredQueryEncodings = torch.from_numpy(requiredQueryEncodings)
            if requiredQueryEncodings.shape[0] > 0:
                requiredQueryLabels = [int(queryLabels[x].item()) for x in range((len(assignments))) if assignments[x] == i]
                trainingParams = {'batch_size': 32}
                trainingDataset = SentenceEncodingDataset(requiredQuerySentences, requiredQueryEncodings, requiredQueryLabels)
                trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
                criterion = self.getCriterion()
                model_1 = supportLines[i].getFirstPrototype().getPrototypeModel()
                model_2 = supportLines[i].getSecondPrototype().getPrototypeModel()
                # put these models on the GPU
                model_1.to(DEVICE)
                model_2.to(DEVICE)
                for j, data in enumerate(trainLoader, 0):
                    if train:
                        # printed = False
                        # print("soft label grads were")
                        # for metaParam in self.softLabelMetaModel.parameters():
                        #     if metaParam.requires_grad and not printed:
                        #         print(metaParam.grad)
                        #         printed = True
                        # get the inputs; data is a list of [inputs, encodings, labels]
                        sentences, encodings, labels = data
                        outputs, distances_1, distances_2 = self.computeLabelsAndDistances(sentences, encodings, model_1, model_2, supportLines[i].getFirstPrototype().getLocation(), supportLines[i].getSecondPrototype().getLocation())
                        # make a copy of the outputs and detach the outputs
                        outputs_optimizer = outputs.clone().detach()
                        # apply a no-op with optimizer_1 and optimizer_2 with detached outputs to add them to the computation graph of optimizers


                        # compute loss with them, do a backward pass and scale the gradients

                        # compute the loss
                        losses_j = criterion(outputs.to(DEVICE), labels.to(DEVICE))
                        outerLoopLoss += losses_j.sum().item()
                        predictions_i = torch.argmax(outputs, dim=1).tolist()
                        labelsList = [labels[i].item() for i in range(labels.shape[0])]
                        outerLoopPredictions.extend(predictions_i)
                        outerLoopLabels.extend(labelsList)
                        self.predictions.extend(predictions_i)
                        self.actualLabels.extend(labelsList)
                        self.losses.extend(losses_j)
                        # calculate the gradients
                        self.manual_backward(losses_j.sum())
                        for name, param in self.optimizer_1.named_parameters():
                            if param.requires_grad:
                                print(name, param.grad)
                        # multiply the calculated gradients of each model by a scaling factor
                        self.updateGradients(losses_j, model_1, model_2, distances_1, distances_2)
                        # in first order approximation, the gradients are the sum of the inner and outer loop models
                        for metaParam, localParam_1, localParam_2 in zip(self.metaLearner.parameters(), model_1.metaLearner.parameters(), model_2.metaLearner.parameters()):
                            if metaParam.requires_grad:
                                if metaParam.grad is None:
                                    metaParam.grad = torch.zeros(localParam_1.grad.shape).to(DEVICE)
                                metaParam.grad += localParam_1.grad
                                metaParam.grad += localParam_2.grad

                        model_1.zero_grad()
                        model_2.zero_grad()

                        # delete the unnecessary objects
                        del losses_j, outputs, distances_1, distances_2
                    else:
                        with torch.no_grad():
                            sentences, encodings, labels = data
                            outputs, _, _ = self.computeLabelsAndDistances(sentences, encodings, model_1, model_2, supportLines[i].getFirstPrototype().getLocation(), supportLines[i].getSecondPrototype().getLocation())
                            # compute the loss
                            losses_j = criterion(outputs.to(DEVICE), labels.to(DEVICE))
                            outerLoopLoss += losses_j.sum().item()
                            predictions_i = torch.argmax(outputs, dim=1).tolist()
                            labels = [labels[i].item() for i in range(labels.shape[0])]
                            outerLoopPredictions.extend(predictions_i)
                            outerLoopLabels.extend(labels)

                            self.predictions.extend(predictions_i)
                            self.actualLabels.extend(labels)
                            self.losses.append(outerLoopLoss)
        if train:
            print("outer loop training accuracy is", accuracy_score(outerLoopLabels, outerLoopPredictions), "and loss is", outerLoopLoss)
            # do a step of the optimizers
        else:
            print("outer loop episodic validation accuracy is", accuracy_score(outerLoopLabels, outerLoopPredictions), "and loss is", outerLoopLoss)
            self.log("outer_loop_validation_loss_" + str(self.val_episode), outerLoopLoss, batch_size=len(outerLoopLabels))
            self.log("outer_loop_validation_accuracy_" + str(self.val_episode), accuracy_score(outerLoopLabels, outerLoopPredictions), batch_size=len(outerLoopLabels))
            self.val_episode += 1
            self.val_episode %= 8  # since we have 8 episodes in a validation set

    def runMetaWorkflow(self, batch, train=True):
        for episode_i in range(len(batch[0])):
            data, labels = batch[0][episode_i], batch[1][episode_i]
            # if the labels are not consistently 0-indexed, remap them for validation loop
            data, labels = self.shuffleAndRemapLabels(data, labels)
            data, labels = self.getSortedEpisode(data, labels)
            # split the data in support and query sets
            supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
            querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
            # compute lines for the support set
            supportEncodings, supportLines = self.computeLines(supportSet, supportLabels)
            queryEncodings, queryLabels = self.getFewShotEncodings(querySet, queryLabels)
            print("Number of labels in the episode are", len(set(supportLabels)), "and lines are", len(supportLines))
            # for each line in the support set, carry out meta-training
            for supportLine in supportLines:
                # do not train if there is only one prototype
                if len(set(supportLine.getLabels())) == 1:
                    continue
                # perform few-shot adaptation on the support set
                self.trainInnerLoop(supportLine, supportSet, supportEncodings, supportLabels)
            # calculate the loss on the query set
            self.runOuterLoop(supportLines, supportEncodings, supportLabels, querySet, queryEncodings, queryLabels, train)
            del supportLines, supportEncodings, queryEncodings
        if train:
            print("outer loop accuracy is", accuracy_score(self.actualLabels, self.predictions), "and total loss is", sum(self.losses).item())
            # normalise the gradients
            for metaParam in self.metaLearner.parameters():
                if metaParam.requires_grad:
                    metaParam.grad = metaParam.grad / len(batch[0])
            # update the soft label model
            self.optimizers().step()
            self.lr_schedulers().step()
            self.optimizers().zero_grad()

    def getSortedEpisode(self, data, labels):
        kShot = labels.count(0)
        supportSet = []
        supportLabels = []
        querySet = []
        queryLabels = []
        for i in range(len(labels)):
            if supportLabels.count(labels[i]) < kShot // 2:
                supportSet.append(data[i])
                supportLabels.append(labels[i])
            else:
                querySet.append(data[i])
                queryLabels.append(labels[i])
        episodeData = supportSet + querySet
        episodeLabels = supportLabels + queryLabels
        return episodeData, episodeLabels

    def shuffleAndRemapLabels(self, data, labels):
        combinedList = list(zip(data, labels))
        random.shuffle(combinedList)
        data, labels = zip(*combinedList)
        data = list(data)
        labels = list(labels)
        labelsDict = {}
        label = 0
        for i in range(len(labels)):
            if labels[i] not in labelsDict:
                labelsDict[labels[i]] = label
                label += 1
        for i in range(len(labels)):
            labels[i] = labelsDict[labels[i]]
        return data, labels

    def computeLines(self, dataset, labels):
        trainingEncodings, trainingLabels = self.getFewShotEncodings(dataset, labels)
        # invoke line generator and compute lines per episode
        trainingSet = {'encodings': trainingEncodings, 'labels': trainingLabels}
        lineGenerator = LineGenerator(trainingSet, PROTOTYPE_META_MODEL)
        lines = lineGenerator.generateLines(self.metaLearner)
        return trainingEncodings, lines

    def getFewShotEncodings(self, data, labels):
        with torch.no_grad():
            training_encodings = []
            training_labels = labels
            for i in range(len(training_labels)):
                encoding = self.metaLearner(data[i]).cpu().detach().reshape(-1)
                training_encodings.append(encoding)
                del encoding
            return torch.stack(training_encodings, dim=0), torch.Tensor(training_labels)

    def validation_step(self, batch, batch_idx):
        self.resetMetrics()
        torch.set_grad_enabled(True)
        self.runMetaWorkflow(batch, train=False)
        torch.set_grad_enabled(False)
        self.log("outer_loop_validation_accuracy", accuracy_score(self.actualLabels, self.predictions), batch_size=len(self.predictions))
        print("validation accuracy for the validation set is", accuracy_score(self.actualLabels, self.predictions), "and the loss is", sum(self.losses), "\n")
        self.log("outer_loop_validation_loss", sum(self.losses), batch_size=len(self.predictions))
        torch.cuda.empty_cache()
        return None

    def training_step(self, batch, batch_idx):
        # zero the meta learning gradients
        self.resetMetrics()
        self.optimizers().zero_grad()
        self.runMetaWorkflow(batch)
        self.log("outer_loop_training_accuracy", accuracy_score(self.actualLabels, self.predictions))
        self.log("outer_loop_training_loss", sum(self.losses).item())
        print("\n")
        torch.cuda.empty_cache()
        return None

    def resetMetrics(self):
        self.actualLabels = []
        self.losses = []
        self.predictions = []
