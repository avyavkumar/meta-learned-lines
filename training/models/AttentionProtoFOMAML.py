import random

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.cuda import OutOfMemoryError
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import higher
from lines.LineGenerator import LineGenerator
from prototypes.models.PrototypeMetaAttentionModel import PrototypeMetaAttentionModel
from utils.Constants import PROTOTYPE_META_MODEL
from utils.ModelUtils import DEVICE

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO check if data is correct and training is going as expected
# TODO log metrics of individual validation episodes
# TODO load model from checkpoint and check if everything works as expected
# TODO write code for distributed hyper param checking

# Adapted from https://github.com/bamos/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
class AttentionProtoFOMAML(L.LightningModule):

    def __init__(self, outerLR, innerLR, attentionLR, steps, batchSize):
        super().__init__()
        self.trainingTaskName = None
        self.metaDataset = None
        self.predictions = []
        self.accuracies = []
        self.actualLabels = []
        self.losses = []
        self.save_hyperparameters()
        self.val_episode = 0
        self.MAX_STEPS = 70
        self.automatic_optimization = False
        self.metaAttentionModel = PrototypeMetaAttentionModel()
        param_groups = [{'params': p, 'lr': innerLR} for p in self.metaAttentionModel.parameters()]
        self.innerLoopOptimiser = torch.optim.SGD(param_groups, lr=innerLR, momentum=0.9, nesterov=True)
        t = higher.optim.get_trainable_opt_params(self.innerLoopOptimiser)
        self.lrs = nn.ParameterList(map(nn.Parameter, t['lr']))
        self.updateOutputLearningRates()
        torch.set_printoptions(threshold=100)
        print("Training AttentionProtoFOMAML with parameters", self.hparams)

    # multiply output learning rates by 1e+1 to initialise their learning at a faster rate
    def updateOutputLearningRates(self):
        outputIndices = []
        for i, (name, param) in enumerate(self.metaAttentionModel.named_parameters()):
            if "linear2" in name or "linear3" in name:
                outputIndices.append(i)
        for index in outputIndices:
            self.lrs[index].data *= 10

    def configure_optimizers(self):
        optimiser = AdamW([{'params': self.lrs},
                           {'params': self.metaAttentionModel.metaLearner.parameters()},
                           {'params': self.metaAttentionModel.linear2_1.parameters(), 'lr': self.hparams.attentionLR},
                           {'params': self.metaAttentionModel.linear2_2.parameters(), 'lr': self.hparams.attentionLR},
                           {'params': self.metaAttentionModel.attention.parameters(), 'lr': self.hparams.attentionLR}], lr=self.hparams.outerLR)
        scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=self.MAX_STEPS, eta_min=1e-5)
        return [optimiser], [scheduler]

    def metaTrainOneEpisode(self, supportSet, supportLabels, querySet, queryLabels, line, validate):
        classes = len(set(supportLabels))
        with higher.innerloop_ctx(self.metaAttentionModel, self.innerLoopOptimiser, copy_initial_weights=validate) as (fnet, diffopt):
            for _ in range(self.hparams.steps):
                supportLogits = fnet(supportSet, supportLabels, line[0], line[-1])
                innerLoss = F.cross_entropy(supportLogits, torch.LongTensor(supportLabels).to(DEVICE))
                diffopt.step(innerLoss, override={'lr': self.lrs})
                accuracy = accuracy_score(supportLabels, torch.argmax(supportLogits, dim=1).tolist())
                print("The inner loop loss is", round(innerLoss.item(), 3), "and the inner accuracy is", round(accuracy, 3))
            outerLogits = fnet.forward_test(supportSet, supportLabels, querySet, classes, line[0], line[-1])
            outerLoss = F.cross_entropy(outerLogits, torch.LongTensor(queryLabels).to(DEVICE))
            accuracy = accuracy_score(queryLabels, torch.argmax(outerLogits, dim=1).tolist())
            # Update the model's meta-parameters to optimize the query set
            # This unrolls through the gradient steps.
            if validate:
                self.manual_backward(outerLoss)
                # dissolve the computational graph to conserve memory but don't take an optimiser step
                self.optimizers().zero_grad()
        return outerLoss, accuracy

    def runMetaWorkflow(self, batch, validate=False):
        for episode_i in range(len(batch[0])):
            self.optimizers().zero_grad(set_to_none=True)
            data, labels = batch[0][episode_i], batch[1][episode_i]
            # if the labels are not consistently 0-indexed, remap them for validation loop
            data, labels = self.shuffleAndRemapLabels(data, labels)
            data, labels = self.getSortedEpisode(data, labels)
            # split the data in support and query sets
            supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
            querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
            # compute lines for the support set
            supportLines = self.computeLines(supportSet, supportLabels)
            if len(supportLines) > 1 and not validate:
                raise RuntimeError("Multiple lines detected during training")
            print("Number of labels in the episode are", len(set(supportLabels)), "and lines are", len(supportLines))
            queryLoss, queryAccuracy = self.metaTrainOneEpisode(supportSet, supportLabels, querySet, queryLabels, supportLines[0], validate)
            print("The outer loop loss is", round(queryLoss.item(), 3), "and accuracy is", round(queryAccuracy, 3))
            self.losses.append(queryLoss)
            self.accuracies.append(queryAccuracy)
        if not validate:
            outerLoss = torch.mean(torch.stack(self.losses))
            self.manual_backward(outerLoss)
            self.optimizers().step()
            if self.current_epoch <= self.MAX_STEPS:
                self.lr_schedulers().step()
            # ensure learning rates are above a threshold value
            for lr in self.lrs:
                lr.data[lr < 1e-6] = 1e-6
            for i in range(len(self.lrs)-12,len(self.lrs)-4):
                print(self.lrs[i])
        # delete objects to preserve memory
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        # zero the meta learning gradients
        self.resetMetrics()
        self.trainingTaskName = batch[2]
        print("Running", self.trainingTaskName + "...")
        self.runMetaWorkflow(batch)
        print("The aggregated outer loop loss is",  round(torch.mean(torch.Tensor(self.losses)).item(), 3), "and accuracy is", round(torch.mean(torch.Tensor(self.accuracies)).item(), 3))
        self.log("outer_loop_training_accuracy_" + self.trainingTaskName, torch.mean(torch.Tensor(self.accuracies)).item())
        self.log("outer_loop_training_loss_" + self.trainingTaskName, torch.mean(torch.Tensor(self.losses)).item())
        print("\n")
        self.trainingTaskName = None
        return None

    def validation_step(self, batch, batch_idx):
        print("Running the validation loop...")
        self.resetMetrics()
        torch.set_grad_enabled(True)
        self.optimizers().zero_grad(set_to_none=True)
        try:
            self.runMetaWorkflow(batch, validate=True)
        except OutOfMemoryError as e:
            print(e)
        finally:
            torch.set_grad_enabled(False)
            self.log("outer_loop_validation_accuracy", torch.mean(torch.Tensor(self.accuracies)).item(), batch_size=len(self.accuracies))
            print("The validation loss is",  round(torch.mean(torch.Tensor(self.losses)).item(), 3), "and validation accuracy is", round(torch.mean(torch.Tensor(self.accuracies)).item(), 3), "\n")
            self.log("outer_loop_validation_loss", torch.mean(torch.Tensor(self.losses)).item(), batch_size=len(self.losses))
            torch.cuda.empty_cache()
            return None

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
        trainingEncodings = self.getFewShotEncodings(dataset)
        # invoke line generator and compute lines per episode
        trainingSet = {'encodings': trainingEncodings, 'labels': torch.LongTensor(labels)}
        lineGenerator = LineGenerator(trainingSet, PROTOTYPE_META_MODEL)
        lines_indices = lineGenerator.generateLineIndices()
        return lines_indices

    def getFewShotEncodings(self, data):
        with torch.no_grad():
            encodings = self.metaAttentionModel.metaLearner(data).cpu().detach()
            return encodings

    def resetMetrics(self):
        self.actualLabels = []
        self.losses = []
        self.predictions = []
        self.accuracies = []
