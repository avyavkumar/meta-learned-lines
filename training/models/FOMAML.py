import random
from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup
from datautils.GLUEEncoderUtils import get_labelled_episodic_training_data
from lines.Line import Line
from lines.lo_shot_utils import dist_to_line_multiD
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils import ModelUtils
from utils.Constants import PROTOTYPE_META_MODEL
from lines.LineGenerator import LineGenerator
from torch.utils.data import DataLoader
import pytorch_lightning as L
from utils.ModelUtils import DEVICE, CPU_DEVICE


# TODO check if data is correct and training is going as expected
# TODO log metrics of individual validation episodes
# TODO load model from checkpoint and check if everything works as expected
# TODO write code for distributed hyper param checking

class FOMAML(L.LightningModule):

    def __init__(self, outerLR, innerLR, outputLR, steps, batchSize, warmupSteps):
        super().__init__()
        self.predictions = []
        self.actualLabels = []
        self.losses = []
        self.save_hyperparameters()
        self.val_episode = 0
        self.automatic_optimization = False
        self.metaLearner = PrototypeMetaModel()
        torch.set_printoptions(threshold=100)
        print("Training FOMAML with parameters", self.hparams)

    def filterEncodingsByLabels(self, labels, training_data, training_labels):
        filteredTrainingData = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingData.append(training_data[i].detach().cpu().numpy())
                filteredTrainingLabels.append(training_labels[i])
        return torch.Tensor(np.array(filteredTrainingData)), np.array(filteredTrainingLabels)

    def configure_optimizers(self):
        optimiser = AdamW(self.metaLearner.parameters(), lr=self.hparams.outerLR)
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": get_constant_schedule_with_warmup(optimizer=optimiser, num_warmup_steps=100)
            }
        }

    def updateGradients(self, losses, model_1, model_2, distances_1, distances_2):
        losses_1 = losses.clone().detach().cpu()
        losses_2 = losses.clone().detach().cpu()
        losses_1 = distances_2.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_1
        losses_2 = distances_1.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_2
        loss_ratio_1 = losses_1.sum() / (losses_1.sum() + losses_2.sum())
        loss_ratio_2 = losses_2.sum() / (losses_1.sum() + losses_2.sum())
        model_1.scaleGradients(loss_ratio_1)
        model_2.scaleGradients(loss_ratio_2)

    def getCriterion(self):
        return nn.CrossEntropyLoss(reduction='none')

    def getInnerLoopOptimiser(self, model):
        return SGD([{'params': model.metaLearner.parameters()},
                    {'params': model.linear.parameters(), 'lr': self.hparams.outputLR}], lr=self.hparams.innerLR)

    def getInnerLoopScheduler(self, optimiser):
        return get_constant_schedule_with_warmup(optimizer=optimiser, num_warmup_steps=self.hparams.warmupSteps)

    def computeLabelsAndDistances(self, inputs, encodings, model_1, model_2, location_1, location_2):
        # output_1 = model_1(inputs).to(CPU_DEVICE)
        # output_2 = model_2(inputs).to(CPU_DEVICE)
        output_1 = model_1(encodings.to(DEVICE)).to(CPU_DEVICE)
        output_2 = model_2(encodings.to(DEVICE)).to(CPU_DEVICE)
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

    def runInnerLoopTrainingStep(self, line, model_1, model_2, optimisers, filteredTrainingSentences, filteredTrainingEncodings,
                                 filteredTrainingLabels, train):
        trainingParams = {
            'batch_size': 32
        }
        trainingDataset = SentenceEncodingDataset(filteredTrainingSentences, filteredTrainingEncodings, filteredTrainingLabels)
        trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
        predictions = []
        correctLabels = []
        criterion = self.getCriterion()
        optimiser_1, optimiser_2, scheduler_1, scheduler_2 = optimisers
        optimiser_1.zero_grad()
        optimiser_2.zero_grad()
        training_losses = []
        for i, data in enumerate(trainLoader, 0):
            # get the inputs; data is a list of [inputs, encodings, labels]
            inputs, encodings, labels = data
            outputs, distances_1, distances_2 = self.computeLabelsAndDistances(inputs, encodings, model_1, model_2,
                                                                               line.getFirstPrototype().getLocation(),
                                                                               line.getSecondPrototype().getLocation())
            # compute the loss
            losses = criterion(outputs, labels)
            predictions_i = torch.argmax(outputs, dim=1).tolist()
            predictions.extend(predictions_i)
            correctLabels.extend(labels)
            if losses.sum().item() > 0:
                # calculate the gradients
                self.manual_backward(losses.sum())
                training_losses.append(losses.sum().item())
                # multiply the calculated gradients of each model by a scaling factor
                self.updateGradients(losses, model_1, model_2, distances_1, distances_2)
                # update the gradients
                optimiser_1.step()
                optimiser_2.step()
                scheduler_1.step()
                scheduler_2.step()
                # zero the parameter gradients
                optimiser_1.zero_grad()
                optimiser_2.zero_grad()
            del outputs, distances_1, distances_2
        print("inner loop training loss is", sum(training_losses), "and accuracy is", accuracy_score(correctLabels, predictions))
        self.log("inner_loop_training_loss", sum(training_losses), batch_size=len(training_losses))
        self.log("inner_loop_training_accuracy", accuracy_score(correctLabels, predictions), batch_size=len(predictions))

    def trainInnerLoop(self, line: Line, supportSet, supportEncodings, supportLabels, train=True):
        # create models for inner loop updates
        innerLoopModel_1 = deepcopy(line.getFirstPrototype().getPrototypeModel())
        innerLoopModel_2 = deepcopy(line.getSecondPrototype().getPrototypeModel())
        # add these models to the GPU if there is a GPU
        innerLoopModel_1.to(ModelUtils.DEVICE)
        innerLoopModel_2.to(ModelUtils.DEVICE)
        # get optimisers
        optimiser_1 = self.getInnerLoopOptimiser(innerLoopModel_1)
        optimiser_2 = self.getInnerLoopOptimiser(innerLoopModel_2)
        scheduler_1 = self.getInnerLoopScheduler(optimiser_1)
        scheduler_2 = self.getInnerLoopScheduler(optimiser_2)
        optimisers = (optimiser_1, optimiser_2, scheduler_1, scheduler_2)
        # filter support encodings and labels to ensure that only line-specific data is used for training
        filteredTrainingSentences, filteredTrainingLabels_1 = self.filterSentencesByLabels(line.getLabels(), supportSet,
                                                                                           supportLabels)
        filteredTrainingEncodings, filteredTrainingLabels_2 = self.filterEncodingsByLabels(line.getLabels(),
                                                                                           supportEncodings,
                                                                                           supportLabels)
        # sanity check to ensure that the filtered data is in the correct order
        assert torch.all(torch.eq(torch.tensor(filteredTrainingLabels_1, dtype=torch.int8),
                                  torch.tensor(filteredTrainingLabels_2, dtype=torch.int8))) == torch.tensor(True)

        # use SGD to carry out few-shot adaptation
        for _ in range(self.hparams.steps):
            self.runInnerLoopTrainingStep(line, innerLoopModel_1, innerLoopModel_2, optimisers, filteredTrainingSentences,
                                          filteredTrainingEncodings, filteredTrainingLabels_2, train)

        # delete everything to free the memory
        del filteredTrainingEncodings
        # put these models on the CPU again as part of the line
        innerLoopModel_1.to(CPU_DEVICE)
        innerLoopModel_2.to(CPU_DEVICE)
        line.getFirstPrototype().setPrototypeModel(innerLoopModel_1)
        line.getSecondPrototype().setPrototypeModel(innerLoopModel_2)

    def runOuterLoop(self, supportLines, querySentences, queryEncodings, queryLabels, train=True):
        assignments = []
        outerLoopLoss = 0.0
        outerLoopPredictions = []
        outerLoopLabels = []
        for point in queryEncodings:
            dists = [
                dist_to_line_multiD(point.detach().cpu().numpy(), line.getFirstPrototype().getLocation().detach().cpu().numpy(),
                                    line.getSecondPrototype().getLocation().detach().cpu().numpy()) for line in supportLines]
            nearest = np.argmin(dists)
            assignments.append(nearest)
        for i in range(len(supportLines)):
            requiredQueryEncodings = np.array([queryEncodings[x].detach().cpu().numpy() for x in range((len(assignments))) if assignments[x] == i])
            requiredQuerySentences = [querySentences[x] for x in range((len(assignments))) if assignments[x] == i]
            requiredQueryEncodings = torch.from_numpy(requiredQueryEncodings)
            if requiredQueryEncodings.shape[0] > 0:
                requiredQueryLabels = [int(queryLabels[x].item()) for x in range((len(assignments))) if assignments[x] == i]
                trainingParams = {
                    'batch_size': 32
                }
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
                        # get the inputs; data is a list of [inputs, encodings, labels]
                        inputs, encodings, labels = data
                        outputs, distances_1, distances_2 = self.computeLabelsAndDistances(inputs, encodings, model_1, model_2,
                                                                                           supportLines[i].getFirstPrototype().getLocation(),
                                                                                           supportLines[i].getSecondPrototype().getLocation())
                        # compute the loss
                        losses_j = criterion(outputs.to(DEVICE), labels.to(DEVICE))
                        outerLoopLoss += losses_j.sum().item()
                        predictions_i = torch.argmax(outputs, dim=1).tolist()
                        labels = [labels[i].item() for i in range(labels.shape[0])]
                        outerLoopPredictions.extend(predictions_i)
                        outerLoopLabels.extend(labels)
                        self.predictions.extend(predictions_i)
                        self.actualLabels.extend(labels)
                        self.losses.extend(losses_j)
                        # calculate the gradients
                        self.manual_backward(losses_j.sum())
                        # multiply the calculated gradients of each model by a scaling factor
                        self.updateGradients(losses_j, model_1, model_2, distances_1, distances_2)
                        # in first order approximation, the gradients are the sum of the inner and outer loop models
                        for metaParam, localParam_1, localParam_2 in zip(self.metaLearner.parameters(),
                                                                         model_1.metaLearner.parameters(),
                                                                         model_2.metaLearner.parameters()):
                            if metaParam.requires_grad:
                                if metaParam.grad is None:
                                    metaParam.grad = torch.zeros(localParam_1.grad.shape).to(DEVICE)
                                metaParam.grad += localParam_1.grad
                                metaParam.grad += localParam_2.grad
                        model_1.zero_grad()
                        model_2.zero_grad()
                        del losses_j, outputs, labels, distances_1, distances_2
                    else:
                        with torch.no_grad():
                            inputs, encodings, labels = data
                            outputs, _, _ = self.computeLabelsAndDistances(inputs, encodings, model_1, model_2,
                                                                           supportLines[i].getFirstPrototype().getLocation(),
                                                                           supportLines[i].getSecondPrototype().getLocation())
                            # compute the loss
                            losses_j = criterion(outputs.to(DEVICE), labels.to(DEVICE))
                            outerLoopLoss += losses_j.sum().item()
                            predictions_i = torch.argmax(outputs, dim=1).tolist()
                            labels = [labels[i].item() for i in range(labels.shape[0])]
                            outerLoopPredictions.extend(predictions_i)
                            outerLoopLabels.extend(labels)
                            self.predictions.extend(predictions_i)
                            self.actualLabels.extend(labels)
                            self.losses.extend(losses_j)
        if train:
            print("outer loop training accuracy is", accuracy_score(outerLoopLabels, outerLoopPredictions), "and loss is", outerLoopLoss)
        else:
            print("outer loop episodic validation accuracy is", accuracy_score(outerLoopLabels, outerLoopPredictions), "and loss is", outerLoopLoss)
            self.log("outer_loop_validation_loss_" + str(self.val_episode), outerLoopLoss, batch_size=len(outerLoopLabels))
            self.log("outer_loop_validation_accuracy_" + str(self.val_episode), accuracy_score(outerLoopLabels, outerLoopPredictions), batch_size=len(outerLoopLabels))
            self.val_episode += 1
            self.val_episode %= 8 # since we have 8 episodes in a validation set

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
            queryEncodings, queryLabels = get_labelled_episodic_training_data(querySet, queryLabels)
            print("Number of labels in the episode are", len(set(supportLabels)), "and lines are", len(supportLines))
            # for each line in the support set, carry out meta-training
            for supportLine in supportLines:
                # do not train if there is only one prototype
                if len(set(supportLine.getLabels())) == 1:
                    continue
                # perform few-shot adaptation on the support set
                self.trainInnerLoop(supportLine, supportSet, supportEncodings, supportLabels, train)
            # calculate the loss on the query set
            self.runOuterLoop(supportLines, querySet, queryEncodings, queryLabels, train)
            del supportLines, supportEncodings, queryEncodings
        if train:
            print("outer loop accuracy is", accuracy_score(self.actualLabels, self.predictions), "and total loss is", sum(self.losses).item())
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
        trainingEncodings, trainingLabels = get_labelled_episodic_training_data(dataset, labels)
        # invoke line generator and compute lines per episode
        trainingSet = {'encodings': trainingEncodings, 'labels': trainingLabels}
        lineGenerator = LineGenerator(trainingSet, PROTOTYPE_META_MODEL)
        lines = lineGenerator.generateLines(self.metaLearner)
        return trainingEncodings, lines

    def validation_step(self, batch, batch_idx):
        self.resetMetrics()
        torch.set_grad_enabled(True)
        self.runMetaWorkflow(batch, train=False)
        torch.set_grad_enabled(False)
        self.log("outer_loop_validation_accuracy", accuracy_score(self.actualLabels, self.predictions), batch_size=len(self.predictions))
        print("validation accuracy for the validation set is", accuracy_score(self.actualLabels, self.predictions), "and the loss is", sum(self.losses), "\n")
        self.log("outer_loop_validation_loss", sum(self.losses).item(), batch_size=len(self.predictions))
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

