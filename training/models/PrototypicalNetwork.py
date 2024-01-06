import copy
import random
from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from transformers import get_constant_schedule_with_warmup
from lines.Line import Line
from lines.lo_shot_utils import dist_to_line_multiD
from prototypes.models.PrototypeEmbedderModel import PrototypeEmbedderModel
from prototypes.models.PrototypeEmbeddingMetaModel import PrototypeEmbeddingMetaModel
from prototypes.models.SoftLabelMetaModel import SoftLabelMetaModel
from prototypes.models.SoftLabelPrototypeMetaModel import SoftLabelPrototypeMetaModel
from training_datasets.EncodingDataset import EncodingDataset
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils import ModelUtils
from utils.Constants import PROTOTYPE_META_MODEL
from lines.LineGenerator import LineGenerator
from torch.utils.data import DataLoader
import pytorch_lightning as L
from utils.ModelUtils import DEVICE, CPU_DEVICE, get_prototypes
import torch.nn.functional as F

class PrototypicalNetwork(L.LightningModule):
    def __init__(self, outerLR):
        super().__init__()
        self.trainingTaskName = None
        self.metaDataset = None
        self.predictions = []
        self.actualLabels = []
        self.losses = []
        self.save_hyperparameters()
        self.val_episode = 0
        self.MAX_STEPS = 1500
        self.prototypeEmbedder = PrototypeEmbedderModel()
        self.validationEpisodes = []
        self.validationLabels = []
        self.validationDataGathered = False
        torch.set_printoptions(threshold=100)
        print("Training PrototypicalNetwork with parameters", self.hparams)

    def criterion(self):
        return nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def configure_optimizers(self):
        optimiser_2 = AdamW(self.prototypeEmbedder.parameters(), lr=self.hparams.outerLR)
        scheduler_2 = CosineAnnealingLR(optimizer=optimiser_2, T_max=self.MAX_STEPS, eta_min=1e-6, verbose=False)
        return [optimiser_2], [scheduler_2]

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

    # def runTrainWorkflow(self, batch):
    #     data, labels = batch[0][0], batch[1][0]
    #     # if the labels are not consistently 0-indexed, remap them for validation loop
    #     data, labels = self.shuffleAndRemapLabels(data, labels)
    #     data, labels = self.getSortedEpisode(data, labels)
    #     # split the data in support and query sets
    #     supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
    #     querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
    #     # calculate anchor from the support points
    #     supportEncodings = self.prototypeEmbedder(supportSet)
    #     anchors, classes = self.calculate_prototypes(supportEncodings, torch.LongTensor(supportLabels).to(DEVICE))
    #     losses = []
    #     criterion = self.criterion()
    #     for class_i in range(classes.shape[0]):
    #         # calculate the positive points from the query set
    #         positive = torch.stack([self.prototypeEmbedder(querySet[i]) for i in range(len(querySet)) if queryLabels[i] == classes[class_i]])
    #         for class_j in range(classes.shape[0]):
    #             if class_j != class_i:
    #                 # calculate the negative points from the query set
    #                 negative = torch.stack([self.prototypeEmbedder(querySet[i]) for i in range(len(querySet)) if queryLabels[i] == classes[class_j]])
    #                 losses.append(criterion(anchors[class_i].unsqueeze(0), positive.squeeze(1), negative.squeeze(1)))
    #     return torch.mean(torch.stack(losses))

    def runTrainWorkflow(self, batch):
        data, labels = batch[0][0], batch[1][0]
        # if the labels are not consistently 0-indexed, remap them for validation loop
        data, labels = self.shuffleAndRemapLabels(data, labels)
        data, labels = self.getSortedEpisode(data, labels)
        # split the data in support and query sets
        supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
        querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
        # calculate anchor from the support points
        supportEncodings = self.prototypeEmbedder(supportSet)
        queryEncodings = self.prototypeEmbedder(querySet)
        anchors, classes = self.calculate_prototypes(supportEncodings, torch.LongTensor(supportLabels).to(DEVICE))
        dist = torch.pow(anchors[None, :] - queryEncodings[:, None], 2).sum(dim=2)  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        correct = torch.LongTensor(queryLabels).to(DEVICE)
        labels = (classes[None, :] == correct[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        loss = F.cross_entropy(preds, labels)
        print("training accuracy is", round(acc.item(), 3), "and loss is", round(loss.item(), 3), "\n")
        return loss

    # from https://pytorch-lightning.readthedocs.io/en/1.7.7/notebooks/course_UvA-DL/12-meta-learning.html
    def calculate_prototypes(self, features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    # from https://pytorch-lightning.readthedocs.io/en/1.7.7/notebooks/course_UvA-DL/12-meta-learning.html
    def runValidationWorkflow(self, data, labels):
        # split the data in support and query sets
        supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
        querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
        supportEncodings = self.prototypeEmbedder(supportSet)
        queryEncodings = self.prototypeEmbedder(querySet)

        anchors, classes = self.calculate_prototypes(supportEncodings, torch.LongTensor(supportLabels).to(DEVICE))

        dist = torch.pow(anchors[None, :] - queryEncodings[:, None], 2).sum(dim=2)  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        correct = torch.LongTensor(queryLabels).to(DEVICE)
        labels = (classes[None, :] == correct[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc

    def validation_step(self, batch, batch_idx):
        self.resetMetrics()
        accuracies = []
        if not self.validationDataGathered:
            for episode_i in range(len(batch[0])):
                data, labels = batch[0][episode_i], batch[1][episode_i]
                # if the labels are not consistently 0-indexed, remap them for validation loop
                data, labels = self.shuffleAndRemapLabels(data, labels)
                data, labels = self.getSortedEpisode(data, labels)
                self.validationEpisodes.append(data)
                self.validationLabels.append(labels)
            self.validationDataGathered = True
        for episode_i in range(len(self.validationEpisodes)):
            _, _, accuracy = self.runValidationWorkflow(self.validationEpisodes[episode_i], self.validationLabels[episode_i])
            accuracies.append(accuracy)
            print("The episodic validation accuracy is", accuracy.item())
        self.log("outer_loop_validation_accuracy", torch.mean(torch.Tensor(accuracies)), batch_size=len(accuracies))
        print("validation accuracy for the validation set is", round(torch.mean(torch.Tensor(accuracies)).item(), 3), "\n")
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        # zero the meta learning gradients
        self.resetMetrics()
        self.trainingTaskName = batch[2]
        print("Running", self.trainingTaskName + "...")
        loss = self.runTrainWorkflow(batch)
        # print("The training loss is", loss.item(), "\n")
        torch.cuda.empty_cache()
        self.trainingTaskName = None
        return loss

    def resetMetrics(self):
        self.actualLabels = []
        self.losses = []
        self.predictions = []