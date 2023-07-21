import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.optim.sgd import SGD
from transformers import get_constant_schedule_with_warmup

from lines.Line import Line
from lines.LineGenerator import LineGenerator
from lines.lo_shot_utils import dist_to_line_multiD
from prototypes.models.PrototypeClassifierModels import CLASSIFIER_MODEL_2NN
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
import pytorch_lightning as L
import torch.nn.functional as F

from training_datasets.EncodingDataset import EncodingDataset
from utils.ModelUtils import get_prototypes, CPU_DEVICE, DEVICE


class ProtoNet(L.LightningModule):

    def __init__(self, metaLearningRate, prototypeLearningRate, steps, batchSize, warmupSteps):
        super().__init__()
        self.metaLearner = PrototypeMetaModel()
        self.save_hyperparameters()
        self.val_episode = 0
        self.actualLabels = []
        self.losses = []
        self.predictions = []

    def configure_optimizers(self):
        optimiser = AdamW(self.metaLearner.parameters(), lr=self.hparams.metaLearningRate)
        scheduler = get_constant_schedule_with_warmup(optimiser, num_warmup_steps=self.hparams.warmupSteps)
        return [optimiser], [scheduler]

    def getCriterion(self):
        return nn.CrossEntropyLoss(reduction='none')

    def getPrototypeOptimiser(self, model):
        return SGD(model.parameters(), lr=self.hparams.prototypeLearningRate)

    def getSortedEpisode(self, batch):
        data, labels = batch
        dataList = [data[i][0] for i in range(len(data))]
        labelsList = labels[0].tolist()
        kShot = labelsList.count(0)
        supportSet = []
        supportLabels = []
        querySet = []
        queryLabels = []
        for i in range(len(labelsList)):
            if supportLabels.count(labelsList[i]) < kShot // 2:
                supportSet.append(dataList[i])
                supportLabels.append(labelsList[i])
            else:
                querySet.append(dataList[i])
                queryLabels.append(labelsList[i])
        episodeData = supportSet + querySet
        episodeLabels = supportLabels + queryLabels
        return episodeData, episodeLabels

    def getPredictions(self, prototypes, prototypeLabels, querySet, queryLabels):
        queryEncodings = self.metaLearner(querySet)
        distances = torch.pow(prototypes[None, :] - queryEncodings[:, None], 2).sum(dim=2)
        predictions = F.log_softmax(-distances, dim=1)
        accuracy = accuracy_score(queryLabels, predictions.argmax(dim=1).detach().cpu().numpy())
        return accuracy, predictions

    def training_step(self, batch, batch_idx):
        episodeData, episodeLabels = self.getSortedEpisode(batch)
        supportSet, supportLabels = episodeData[0:len(episodeData) // 2], episodeLabels[0:len(episodeData) // 2]
        supportEncodings = self.metaLearner(supportSet)
        querySet, queryLabels = episodeData[len(episodeData) // 2:], episodeLabels[len(episodeData) // 2:]
        prototypes, prototypeLabels = get_prototypes(supportEncodings, supportLabels)
        accuracy, predictions = self.getPredictions(prototypes, prototypeLabels, querySet, queryLabels)
        loss = F.cross_entropy(predictions, torch.Tensor(queryLabels).long().to(DEVICE))
        print("The training loss is", round(loss.item(), 2), "and the training accuracy is", round(accuracy, 2))
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        return loss

    def resetMetrics(self):
        self.actualLabels = []
        self.losses = []
        self.predictions = []

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.resetMetrics()
        for episode_i in range(len(batch[0])):
            data, labels = batch[0][episode_i], batch[1][episode_i]
            # if the labels are not consistently 0-indexed, remap them for validation loop
            labels = self.remapLabels(labels)
            # split the data in support and query sets
            supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
            querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
            # compute lines for the support set
            supportEncodings, supportLines = self.computeLines(supportSet, supportLabels)
            queryEncodings, queryLabels = self.metaLearner(querySet), torch.Tensor(queryLabels)
            print("Number of labels in the episode are", len(set(supportLabels)), "and lines are", len(supportLines))
            # for each line in the support set, carry out meta-training
            for supportLine in supportLines:
                # do not train if there is only one prototype
                if len(set(supportLine.getLabels())) == 1:
                    continue
                # perform few-shot adaptation on the support set
                self.fineTunePrototypes(supportLine, supportEncodings, supportLabels)
            # calculate the loss on the query set
            self.checkPrototypePerformance(supportLines, queryEncodings, queryLabels)
            del supportLines, supportEncodings, queryEncodings
        torch.set_grad_enabled(False)
        print("The validation loss for the set is ", round(sum(self.losses).item(), 2), "and accuracy is",  round(accuracy_score(self.actualLabels, self.predictions), 2), "\n")
        self.log("outer_loop_validation_accuracy", accuracy_score(self.actualLabels, self.predictions), batch_size=len(self.predictions))
        self.log("outer_loop_validation_loss", sum(self.losses), batch_size=len(self.predictions))
        return sum(self.losses)

    def checkPrototypePerformance(self, supportLines, queryEncodings, queryLabels):
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
            requiredQueryEncodings = torch.from_numpy(requiredQueryEncodings)
            if requiredQueryEncodings.shape[0] > 0:
                requiredQueryLabels = [int(queryLabels[x].item()) for x in range((len(assignments))) if assignments[x] == i]
                trainingParams = {
                    'batch_size': self.hparams.batchSize
                }
                trainingDataset = EncodingDataset(requiredQueryEncodings, requiredQueryLabels)
                trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
                criterion = self.getCriterion()
                model_1 = supportLines[i].getFirstPrototype().getPrototypeModel()
                model_2 = supportLines[i].getSecondPrototype().getPrototypeModel()
                # put these models on the GPU
                model_1.to(DEVICE)
                model_2.to(DEVICE)
                for j, data in enumerate(trainLoader, 0):
                    with torch.no_grad():
                        encodings, labels = data
                        outputs, distances_1, distances_2 = self.computeLabelsAndDistances(encodings, model_1, model_2,
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
            print("outer loop episodic validation accuracy is", round(accuracy_score(outerLoopLabels, outerLoopPredictions), 2), "and loss is", round(outerLoopLoss, 2))
            self.log("outer_loop_validation_loss_" + str(self.val_episode), outerLoopLoss, batch_size=len(outerLoopLabels))
            self.log("outer_loop_validation_accuracy_" + str(self.val_episode), accuracy_score(outerLoopLabels, outerLoopPredictions), batch_size=len(outerLoopLabels))
            self.val_episode += 1
            self.val_episode %= 8 # since we have 8 episodes in a validation set

    def fineTunePrototypes(self, line: Line, supportEncodings, supportLabels):
        # filter support encodings and labels to ensure that only line-specific data is used for training
        filteredTrainingEncodings, filteredTrainingLabels = self.filterEncodingsByLabels(line.getLabels(), supportEncodings, supportLabels)

        # use SGD to carry out few-shot adaptation
        for _ in range(self.hparams.steps):
            self.runInnerLoopTrainingStep(line, filteredTrainingEncodings, filteredTrainingLabels)

    def runInnerLoopTrainingStep(self, line, filteredTrainingEncodings, filteredTrainingLabels):
        trainingParams = {
            'batch_size': self.hparams.batchSize
        }
        model_1 = line.getFirstPrototype().getPrototypeModel()
        model_2 = line.getSecondPrototype().getPrototypeModel()
        trainingDataset = EncodingDataset(filteredTrainingEncodings, filteredTrainingLabels)
        trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)
        predictions = []
        correctLabels = []
        criterion = self.getCriterion()
        optimiser_1 = self.getPrototypeOptimiser(model_1)
        optimiser_2 = self.getPrototypeOptimiser(model_2)
        optimiser_1.zero_grad()
        optimiser_2.zero_grad()
        training_losses = []
        model_1.to(DEVICE)
        model_2.to(DEVICE)
        for i, data in enumerate(trainLoader, 0):
            # get the inputs; data is a list of [inputs, encodings, labels]
            encodings, labels = data
            outputs, distances_1, distances_2 = self.computeLabelsAndDistances(encodings, model_1, model_2,
                                                                               line.getFirstPrototype().getLocation(),
                                                                               line.getSecondPrototype().getLocation())
            # compute the loss
            losses = criterion(outputs, labels)
            predictions_i = torch.argmax(outputs, dim=1).tolist()
            predictions.extend(predictions_i)
            correctLabels.extend(labels)
            if losses.sum().item() > 0:
                # calculate the gradients
                losses.sum().backward()
                training_losses.append(losses.sum().item())
                # multiply the calculated gradients of each model by a scaling factor
                self.updateGradients(losses, model_1, model_2, distances_1, distances_2)
                # update the gradients
                optimiser_1.step()
                optimiser_2.step()
                # zero the parameter gradients
                optimiser_1.zero_grad()
                optimiser_2.zero_grad()
            del outputs, distances_1, distances_2
        print("prototype training loss is", round(sum(training_losses), 2), "and accuracy is", round(accuracy_score(correctLabels, predictions), 2))

    def updateGradients(self, losses, model_1, model_2, distances_1, distances_2):
        losses_1 = losses.clone().detach().cpu()
        losses_2 = losses.clone().detach().cpu()
        losses_1 = distances_2.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_1
        losses_2 = distances_1.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_2
        loss_ratio_1 = losses_1.sum() / (losses_1.sum() + losses_2.sum())
        loss_ratio_2 = losses_2.sum() / (losses_1.sum() + losses_2.sum())
        model_1.scaleGradients(loss_ratio_1)
        model_2.scaleGradients(loss_ratio_2)

    def computeLabelsAndDistances(self, encodings, model_1, model_2, location_1, location_2):
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

    def remapLabels(self, labels):
        sortedDistinctLabels = sorted(set(labels))
        if sortedDistinctLabels != list(range(len(sortedDistinctLabels))):
            labelsDict = {}
            label = 0
            for i in range(len(labels)):
                if labels[i] not in labelsDict:
                    labelsDict[labels[i]] = label
                    label += 1
            for i in range(len(labels)):
                labels[i] = labelsDict[labels[i]]
        return labels

    def filterSentencesByLabels(self, labels, training_data, training_labels):
        filteredTrainingSentences = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingSentences.append(training_data[i])
                filteredTrainingLabels.append(training_labels[i])
        return filteredTrainingSentences, np.array(filteredTrainingLabels)

    def filterEncodingsByLabels(self, labels, training_data, training_labels):
        filteredTrainingData = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingData.append(training_data[i].detach().cpu().numpy())
                filteredTrainingLabels.append(training_labels[i])
        return torch.Tensor(np.array(filteredTrainingData)), np.array(filteredTrainingLabels)

    def computeLines(self, dataset, labels):
        trainingEncodings, trainingLabels = self.metaLearner(dataset), torch.Tensor(labels)
        # invoke line generator and compute lines per episode
        trainingSet = {'encodings': trainingEncodings, 'labels': trainingLabels}
        lineGenerator = LineGenerator(trainingSet, CLASSIFIER_MODEL_2NN)
        lines = lineGenerator.generateLines(self.metaLearner)
        return trainingEncodings, lines
