import random
from copy import deepcopy

import numpy as np
import pytorch_lightning as L
import torch.nn.functional as F

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup

from datautils.LEOPARDEncoderUtils import get_labelled_centroids
from lines.Line import Line
from lines.LineGenerator import LineGenerator
from lines.lo_shot_utils import dist_to_line_multiD
from prototypes.models.PrototypeMetaLinearModel import PrototypeMetaLinearUnifiedModel
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from training_datasets.SentenceDataset import SentenceDataset
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils import ModelUtils
from utils.Constants import PROTOTYPE_META_MODEL
from utils.ModelUtils import DEVICE, CPU_DEVICE


# TODO check if data is correct and training is going as expected
# TODO log metrics of individual validation episodes
# TODO load model from checkpoint and check if everything works as expected
# TODO write code for distributed hyper param checking

class ProtoFOMAML(L.LightningModule):

    def __init__(self, outerLR, innerLR, outputLR, steps, batchSize, warmupSteps):
        super().__init__()
        self.trainingTaskName = None
        self.metaDataset = None
        self.predictions = []
        self.actualLabels = []
        self.losses = []
        self.save_hyperparameters()
        self.val_episode = 0
        self.MAX_STEPS = 100000
        self.automatic_optimization = False
        self.metaLearner = PrototypeMetaModel()
        torch.set_printoptions(threshold=100)
        self.batch = None
        print("Training ProtoFOMAML with parameters", self.hparams)

    def filterEncodingsByLabels(self, labels, training_data, training_labels):
        filteredTrainingData = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingData.append(training_data[i].detach().cpu().numpy())
                filteredTrainingLabels.append(training_labels[i])
        return torch.Tensor(np.array(filteredTrainingData)), np.array(filteredTrainingLabels)

    def configure_optimizers(self):
        optimiser_2 = AdamW(self.metaLearner.parameters(), lr=self.hparams.outerLR)
        scheduler_2 = MultiStepLR(optimiser_2, milestones=[self.MAX_STEPS], gamma=0.1)
        optimiser_3 = AdamW(self.metaLearner.parameters(), lr=self.hparams.outerLR)
        scheduler_3 = MultiStepLR(optimiser_3, milestones=[self.MAX_STEPS], gamma=0.1)
        return [optimiser_2, optimiser_3], [scheduler_2, scheduler_3]

    def updateGradients(self, losses, model, distances_1, distances_2):
        losses_1 = losses.clone().detach().cpu()
        losses_2 = losses.clone().detach().cpu()
        losses_1 = distances_2.to(CPU_DEVICE).squeeze(1) / torch.sum(torch.cat((distances_1.to(CPU_DEVICE), distances_2.to(CPU_DEVICE)), 1), dim=1) * losses_1
        losses_2 = distances_1.to(CPU_DEVICE).squeeze(1) / torch.sum(torch.cat((distances_1.to(CPU_DEVICE), distances_2.to(CPU_DEVICE)), 1), dim=1) * losses_2
        loss_ratio_1 = losses_1.sum() / (losses_1.sum() + losses_2.sum())
        loss_ratio_2 = losses_2.sum() / (losses_1.sum() + losses_2.sum())
        model.scaleModelGradients(loss_ratio_1, loss_ratio_2)

    def getCriterion(self):
        return nn.CrossEntropyLoss(reduction='none')

    def getInnerLoopOptimiser(self, model, classes, train=False):
        return SGD([{'params': model.metaLearner.parameters(), 'lr': self.hparams.innerLR},
                        {'params': model.linear_1.parameters(), 'lr': self.hparams.innerLR},
                        {'params': model.linear_2.parameters(), 'lr': self.hparams.innerLR}], momentum=0.9, nesterov=True)

    def filterSentencesByLabels(self, labels, training_data, training_labels):
        filteredTrainingSentences = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingSentences.append(training_data[i])
                filteredTrainingLabels.append(training_labels[i])
        return filteredTrainingSentences, np.array(filteredTrainingLabels)

    def getFewShotPrototypicalEncodings(self, data, labels, model):
        with torch.no_grad():
            training_encodings = []
            training_labels = labels
            for i in range(len(training_labels)):
                encoding = model.metaLearner(data[i]).cpu().detach().reshape(-1)
                training_encodings.append(encoding)
                del encoding
            return torch.stack(training_encodings, dim=0), torch.LongTensor(training_labels)

    def runInnerLoopTrainingStep(self, line, model, optimiser, filteredTrainingSentences, filteredTrainingLabels, train):
        trainingParams = {'batch_size': 64}
        trainingDataset = SentenceDataset(filteredTrainingSentences, filteredTrainingLabels)
        trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
        predictions = []
        correctLabels = []
        criterion = self.getCriterion()
        optimiser.zero_grad()
        training_losses = []
        for i, data in enumerate(trainLoader, 0):
            # get the inputs; data is a list of [inputs, encodings, labels]
            inputs, labels = data
            # the encodings are derived from prototype 1 and prototype 2 for each class
            outputs, distances_1, distances_2 = model(inputs, labels, line[0], line[-1])
            # compute the loss
            losses = criterion(outputs, labels.to(DEVICE))
            predictions_i = torch.argmax(outputs, dim=1).tolist()
            predictions.extend(predictions_i)
            correctLabels.extend(labels)
            if losses.mean().item() > 0:
                # calculate the gradients
                self.manual_backward(losses.mean())
                training_losses.append(losses.mean().item())
                # multiply the calculated gradients of each model by a scaling factor
                self.updateGradients(losses, model, distances_1, distances_2)
                # update the gradients
                optimiser.step()
                # zero the parameter gradients
                optimiser.zero_grad()
            del outputs, distances_1, distances_2
        print("inner loop training loss is", np.mean(training_losses), "and accuracy is", accuracy_score(correctLabels, predictions))

    def trainInnerLoop(self, line, supportSet, supportLabels, train=True):
        # create the unified model with a clone of the meta learner
        metaLearnerClone = deepcopy(self.metaLearner)
        innerModel = PrototypeMetaLinearUnifiedModel(metaLearnerClone, classes=len(set(supportLabels)), protoFOMAML=True)
        # add these models to the GPU if there is a GPU
        innerModel.to(ModelUtils.DEVICE)
        classes = len(set(supportLabels))
        # get optimisers
        optimiser = self.getInnerLoopOptimiser(innerModel, classes, train)
        # filter support encodings and labels to ensure that only line-specific data is used for training
        filteredTrainingSentences, filteredTrainingLabels = self.filterSentencesByLabels(line, supportSet, supportLabels)
        with torch.no_grad():
            encodings_1 = innerModel.metaLearner(filteredTrainingSentences)
        # use SGD to carry out few-shot adaptation
        for _ in range(self.hparams.steps):
            self.runInnerLoopTrainingStep(line, innerModel, optimiser, filteredTrainingSentences, filteredTrainingLabels, train)
        with torch.no_grad():
            encodings_2 = innerModel.metaLearner(filteredTrainingSentences)
        if not train:
            meanEncodingSimilarity = torch.mean(torch.cosine_similarity(encodings_1, encodings_2, dim=0))
            print("Similarity in encodings is", meanEncodingSimilarity.item())
        # put these models on the CPU again as part of the line
        innerModel.to(CPU_DEVICE)
        return innerModel

    def runOuterLoop(self, supportLines, supportSet, supportLabels, innerModel, querySentences, queryLabels, train=True):
        outerLoopLoss = 0.0
        outerLoopPredictions = []
        outerLoopLabels = []
        # put these models on the GPU
        innerModel.to(DEVICE)
        if train:
            for i in range(len(supportLines)):
                queryEncodings, queryLabels = self.getFewShotPrototypicalEncodings(querySentences, queryLabels, innerModel)
                trainingParams = {'batch_size': 64}
                trainingDataset = SentenceEncodingDataset(querySentences, queryEncodings, queryLabels)
                trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
                criterion = self.getCriterion()
                for j, data in enumerate(trainLoader, 0):
                    sentences, encodings, labels = data
                    outputs = innerModel.forward_test(supportSet, supportLabels, querySentences, supportLines[i][0], supportLines[i][-1])
                    # compute the loss
                    losses_j = criterion(outputs.to(DEVICE), labels.to(DEVICE))
                    # calculate the gradients
                    self.manual_backward(losses_j.mean())
                    outerLoopLoss += losses_j.sum().item()
                    predictions_i = torch.argmax(outputs, dim=1).tolist()
                    labels = [labels[i].item() for i in range(labels.shape[0])]
                    outerLoopPredictions.extend(predictions_i)
                    outerLoopLabels.extend(labels)
                    for metaParam, localParam in zip(self.metaLearner.parameters(), innerModel.metaLearner.parameters()):
                        if metaParam.requires_grad:
                            if metaParam.grad is None:
                                metaParam.grad = torch.zeros(localParam.grad.shape).to(DEVICE)
                            metaParam.grad += localParam.grad
                    self.predictions.extend(predictions_i)
                    self.actualLabels.extend(labels)
                    self.losses.append(outerLoopLoss)
            print("outer loop training accuracy is", round(accuracy_score(outerLoopLabels, outerLoopPredictions), 3), "and loss is", round(outerLoopLoss, 3))
        else:
            for i in range(len(supportLines)):
                queryEncodings, queryLabels = self.getFewShotPrototypicalEncodings(querySentences, queryLabels, innerModel)
                trainingParams = {'batch_size': 64}
                trainingDataset = SentenceEncodingDataset(querySentences, queryEncodings, queryLabels)
                trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
                criterion = self.getCriterion()
                for j, data in enumerate(trainLoader, 0):
                    with torch.no_grad():
                        sentences, encodings, labels = data
                        outputs = innerModel.forward_test(supportSet, supportLabels, querySentences, supportLines[i][0], supportLines[i][-1])
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
                print("outer loop episodic validation accuracy is", accuracy_score(outerLoopLabels, outerLoopPredictions), "and loss is", outerLoopLoss)
                self.log("outer_loop_validation_loss_" + str(self.val_episode), outerLoopLoss, batch_size=len(outerLoopLabels))
                self.log("outer_loop_validation_accuracy_" + str(self.val_episode), accuracy_score(outerLoopLabels, outerLoopPredictions), batch_size=len(outerLoopLabels))
                self.val_episode += 1
                self.val_episode %= 3  # since we have 3 episodes in a validation set

    def runTrainingWorkflow(self, batch):
        for episode_i in range(len(batch[0])):
            data, labels = batch[0][episode_i], batch[1][episode_i]
            # if the labels are not consistently 0-indexed, remap them
            data, labels = self.shuffleAndRemapLabels(data, labels)
            supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
            querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
            # compute lines for the support set
            _, supportLines = self.computeLines(supportSet, supportLabels)
            if len(supportLines) > 1:
                raise RuntimeError("Multiple lines detected during training")
            print("Number of labels in the episode are", len(set(labels)), "and lines are", len(supportLines))
            innerModel = None
            # for each line in the support set, carry out meta-training
            for supportLine in supportLines:
                # do not train if there is only one prototype
                if len(set(supportLine)) == 1:
                    continue
                # perform few-shot adaptation on the support set
                innerModel = self.trainInnerLoop(supportLine, data, labels, train=True)
            # calculate the loss on the query set
            self.runOuterLoop(supportLines, supportSet, supportLabels, innerModel, querySet, queryLabels, train=True)
        print("Updating parameters...")
        classes = len(set(batch[1][0]))
        # normalise the gradients
        for metaParam in self.metaLearner.parameters():
            if metaParam.requires_grad:
                metaParam.grad = metaParam.grad / len(batch[0])
        if classes == 2:
            # update the soft label model
            self.optimizers()[0].step()
            self.lr_schedulers()[0].step()
            self.optimizers()[0].zero_grad()
        else:
            self.optimizers()[1].step()
            self.lr_schedulers()[1].step()
            self.optimizers()[1].zero_grad()
        del supportLines, innerModel

    def runValidationMetaWorkflow(self):
        for episode_i in range(len(self.batch[0])):
            data, labels = self.batch[0][episode_i], self.batch[1][episode_i]
            # split the data in support and query sets for validation
            supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
            querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
            # compute lines for the support set
            _, supportLines = self.computeLines(supportSet, supportLabels)
            print("Number of labels in the episode are", len(set(supportLabels)), "and lines are", len(supportLines))
            innerModel = None
            # for each line in the support set, carry out meta-training
            for supportLine in supportLines:
                # do not train if there is only one prototype
                if len(set(supportLine)) == 1:
                    continue
                # perform few-shot adaptation on the support set
                innerModel = self.trainInnerLoop(supportLine, supportSet, supportLabels, train=False)
            # calculate the loss on the query set
            self.runOuterLoop(supportLines, supportSet, supportLabels, innerModel, querySet, queryLabels, train=False)
            del supportLines, innerModel

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
        lines = lineGenerator.generateLineIndices()
        return trainingEncodings, lines

    def getFewShotEncodings(self, data, labels):
        with torch.no_grad():
            training_encodings = []
            training_labels = labels
            for i in range(len(training_labels)):
                encoding = self.metaLearner(data[i]).cpu().detach().reshape(-1)
                training_encodings.append(encoding)
                del encoding
            return torch.stack(training_encodings, dim=0), torch.LongTensor(training_labels)

    def validation_step(self, batch, batch_idx):
        if self.batch is None:
            self.batch = batch
            for episode_i in range(len(self.batch[0])):
                data, labels = self.batch[0][episode_i], self.batch[1][episode_i]
                # if the labels are not consistently 0-indexed, remap them for validation loop
                data, labels = self.shuffleAndRemapLabels(data, labels)
                data, labels = self.getSortedEpisode(data, labels)
                self.batch[0][episode_i] = data
                self.batch[1][episode_i] = labels
        self.resetMetrics()
        torch.set_grad_enabled(True)
        self.runValidationMetaWorkflow()
        torch.set_grad_enabled(False)
        self.log("outer_loop_validation_accuracy", accuracy_score(self.actualLabels, self.predictions), batch_size=len(self.predictions))
        print("validation accuracy for the validation set is", accuracy_score(self.actualLabels, self.predictions), "and the loss is", sum(self.losses), "\n")
        self.log("outer_loop_validation_loss", sum(self.losses), batch_size=len(self.predictions))
        torch.cuda.empty_cache()
        return None

    def training_step(self, batch, batch_idx):
        # zero the meta learning gradients
        self.resetMetrics()
        for optimiser in self.optimizers():
            optimiser.zero_grad()
        self.trainingTaskName = batch[2]
        print("Running", self.trainingTaskName + "...")
        self.runTrainingWorkflow(batch)
        print("aggregated outer loop accuracy is", accuracy_score(self.actualLabels, self.predictions), "and the loss is", sum(self.losses), "\n")
        self.log("outer_loop_training_accuracy_" + self.trainingTaskName, accuracy_score(self.actualLabels, self.predictions))
        self.log("outer_loop_training_loss_" + self.trainingTaskName, sum(self.losses))
        torch.cuda.empty_cache()
        self.trainingTaskName = None
        return None

    def resetMetrics(self):
        self.actualLabels = []
        self.losses = []
        self.predictions = []
