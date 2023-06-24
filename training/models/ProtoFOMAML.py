from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn
from torch.optim import SGD
from transformers import AdamW, get_constant_schedule_with_warmup
from datautils.GLUEEncoderUtils import get_labelled_GLUE_episodic_training_data
from lines.Line import Line
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils.Constants import PROTOTYPE_META_MODEL
from lines.LineGenerator import LineGenerator
from torch.utils.data import DataLoader
import pytorch_lightning as L
from utils.ModelUtils import get_prototypes


# TODO save the best model
# TODO implement early stopping#
# TODO check if the training is happening as expected - meta-layer is consistent across all layers
class ProtoFOMAML(L.LightningModule):

    def __init__(self, outerLR, innerLR, outputLR, steps, batchSize, warmupSteps):
        super().__init__()
        self.save_hyperparameters()
        self.metaLearner = PrototypeMetaModel()

    def filterEncodingsByLabels(self, labels, training_data, training_labels):
        filteredTrainingData = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingData.append(training_data[i].detach().numpy())
                filteredTrainingLabels.append(training_labels[i])
        return torch.Tensor(np.array(filteredTrainingData)), np.array(filteredTrainingLabels)

    def getPredictions(self, outputs, line):
        predictedLabels = []
        argmaxLabels = torch.argmax(outputs, dim=1).tolist()
        reverseLookup = {}
        for key, value in line.getLabelDict().items():
            reverseLookup[value] = key
        for i in range(len(argmaxLabels)):
            predictedLabels.append(reverseLookup[argmaxLabels[i]])
        return predictedLabels

    def updateGradients(self, losses, model_1, model_2, distances_1, distances_2):
        """
        The pseudocode for scaling gradients is as follows
            for each loss
              divide the loss by the sum of the distances
              compute loss for the first model by multiplying loss*(distance_2/(distance_1 + distance_2))
              compute loss for the second model by multiplying loss*(distance_2/(distance_1 + distance_2))
            get the ratio of the losses
            multiply the gradients by the factor required
        :param losses: the losses per mini-batch
        :param prototype_1: the first soft-label prototype in the line
        :param prototype_2: the second soft-label prototype in the line
        :param distances_1: the distances from the first soft-label prototype in the mini-batch
        :param distances_2: the distances from the second soft-label prototype in the mini-batch
        """
        losses_1 = losses.clone().detach()
        losses_2 = losses.clone().detach()

        losses_1 = distances_2.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_1
        losses_2 = distances_1.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_2

        loss_ratio_1 = (losses_1.sum() / (losses_1.sum() + losses_2.sum()))
        loss_ratio_2 = (losses_2.sum() / (losses_1.sum() + losses_2.sum()))

        model_1.scaleGradients(loss_ratio_1)
        model_2.scaleGradients(loss_ratio_2)

    def convertLabels(self, labelDict, labels):
        argLabels = []
        for i in range(len(labels)):
            argLabels.append(labelDict[labels[i].item()])
        return torch.Tensor(argLabels).long()

    def getCriterion(self):
        return nn.CrossEntropyLoss(reduction='none')

    def getInnerLoopOptimiser(self, model):
        return SGD([{'params': model.metaLearner.parameters()},
                    {'params': model.linear.parameters(), 'lr': self.hparams.outputLR}], lr=self.hparams.innerLR)

    def getMetaLearningOptimiser(self, model):
        return AdamW(model.parameters(), lr=self.hparams.outerLR)

    def getLearningRateScheduler(self, optimiser):
        return get_constant_schedule_with_warmup(optimiser, num_warmup_steps=self.hparams.warmupSteps)

    def computeLabelsAndDistances(self, inputs, encodings, model_1, model_2, location_1, location_2):

        output_1 = model_1(inputs)
        output_2 = model_2(inputs)

        # get distances from the prototypes for all inputs
        distances_1 = []
        distances_2 = []
        for i in range(encodings.shape[0]):
            distances_1.append(np.linalg.norm(encodings[i].detach().numpy() - location_1.detach().numpy()))
            distances_2.append(np.linalg.norm(encodings[i].detach().numpy() - location_2.detach().numpy()))

        distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1)
        distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1)

        # compute the weighted probability distribution
        outputs = output_1 / distances_1 + output_2 / distances_2

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

    def runInnerLoopTrainingStep(self, line, model_1, model_2, filteredTrainingSentences, filteredTrainingEncodings,
                                 filteredTrainingLabels):

        # TODO instantiate the training params
        trainingParams = {
            'batch_size': self.hparams.batchSize
        }

        trainingDataset = SentenceEncodingDataset(filteredTrainingSentences, filteredTrainingEncodings,
                                                  filteredTrainingLabels)
        trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)

        criterion = self.getCriterion()
        optimiser_1 = self.getInnerLoopOptimiser(model_1)
        optimiser_2 = self.getInnerLoopOptimiser(model_2)
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
            print("losses are", losses, "and loss is", losses.sum().item())
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

    def trainInnerLoop(self, line: Line, supportSet, supportEncodings, supportLabels):
        # create models for inner loop updates
        innerLoopModel_1 = deepcopy(line.getFirstPrototype().getPrototypeModel())
        innerLoopModel_2 = deepcopy(line.getSecondPrototype().getPrototypeModel())

        # filter support encodings and labels to ensure that only line-specific data is used for training
        filteredTrainingSentences, filteredTrainingLabels_1 = self.filterSentencesByLabels(line.getLabels(), supportSet,
                                                                                           supportLabels)
        filteredTrainingEncodings, filteredTrainingLabels_2 = self.filterEncodingsByLabels(line.getLabels(),
                                                                                           supportEncodings,
                                                                                           supportLabels)

        # sanity check to ensure that the filtered data is in the correct order
        assert torch.all(torch.eq(torch.tensor(filteredTrainingLabels_1, dtype=torch.int8),
                                  torch.tensor(filteredTrainingLabels_2, dtype=torch.int8))) == torch.tensor(True)

        filteredTrainingLabels = [line.getLabelDict()[label] for label in filteredTrainingLabels_1]

        prototypicalEmbeddings_1 = innerLoopModel_1.getPrototypicalEmbedding(filteredTrainingSentences)
        prototypicalEmbeddings_2 = innerLoopModel_2.getPrototypicalEmbedding(filteredTrainingSentences)

        # sanity check to ensure that both models are initialised the same way
        assert torch.equal(prototypicalEmbeddings_1, prototypicalEmbeddings_2)

        # calculate prototypes and use them to initialise the weights of the linear layer
        prototypes, _ = get_prototypes(prototypicalEmbeddings_1, filteredTrainingLabels)

        # TODO the biases are very high!
        innerLoopModel_1.setParamsOfLinearLayer(2 * prototypes, -torch.norm(prototypes, dim=1) ** 2)
        innerLoopModel_2.setParamsOfLinearLayer(2 * prototypes, -torch.norm(prototypes, dim=1) ** 2)

        # use SGD to carry out few-shot adaptation
        for _ in range(self.hparams.steps):
            self.runInnerLoopTrainingStep(line, innerLoopModel_1, innerLoopModel_2, filteredTrainingSentences,
                                          filteredTrainingEncodings, filteredTrainingLabels)

        return innerLoopModel_1, innerLoopModel_2

    def trainOuterLoop(self, line, model_1, model_2, querySet, queryEncodings, queryLabels):

        # filter support encodings and labels to ensure that only line-specific data is used for training
        filteredTrainingSentences, filteredTrainingLabels_1 = self.filterSentencesByLabels(line.getLabels(), querySet,
                                                                                           queryLabels)
        filteredTrainingEncodings, filteredTrainingLabels_2 = self.filterEncodingsByLabels(line.getLabels(),
                                                                                           queryEncodings,
                                                                                           queryLabels)

        # sanity check to ensure that the filtered data is in the correct order
        assert torch.all(torch.eq(torch.tensor(filteredTrainingLabels_1, dtype=torch.int8),
                                  torch.tensor(filteredTrainingLabels_2, dtype=torch.int8))) == torch.tensor(True)

        filteredTrainingLabels = [line.getLabelDict()[label] for label in filteredTrainingLabels_1]

        # run the model and get the outputs from the query set
        # TODO instantiate the training params
        trainingParams = {
            'batch_size': self.hparams.batchSize
        }

        trainingDataset = SentenceEncodingDataset(filteredTrainingSentences, filteredTrainingEncodings,
                                                  filteredTrainingLabels)
        trainLoader = torch.utils.data.DataLoader(trainingDataset, **trainingParams)

        criterion = self.getCriterion()
        optimiser_1 = self.getMetaLearningOptimiser(line.getFirstPrototype().getPrototypeModel())
        scheduler_1 = self.getLearningRateScheduler(optimiser_1)
        optimiser_2 = self.getMetaLearningOptimiser(line.getSecondPrototype().getPrototypeModel())
        scheduler_2 = self.getLearningRateScheduler(optimiser_2)

        # zero the parameter gradients
        optimiser_1.zero_grad()
        optimiser_2.zero_grad()

        for i, data in enumerate(trainLoader, 0):

            # get the inputs; data is a list of [inputs, encodings, labels]
            inputs, encodings, labels = data

            outputs, distances_1, distances_2 = self.computeLabelsAndDistances(inputs, encodings, model_1, model_2,
                                                                               line.getFirstPrototype().getLocation(),
                                                                               line.getSecondPrototype().getLocation())

            # compute the loss
            losses = criterion(outputs, labels)
            print("outer loop losses are", losses, "and loss is", losses.sum().item())
            # calculate the gradients
            losses.sum().backward()

            # multiply the calculated gradients of each model by a scaling factor
            self.updateGradients(losses, model_1, model_2, distances_1, distances_2)

            for metaParam_1, metaParam_2, localParam_1, localParam_2 in zip(
                    line.getFirstPrototype().getPrototypeModel().metaLearner.parameters(),
                    line.getSecondPrototype().getPrototypeModel().metaLearner.parameters(),
                    model_1.metaLearner.parameters(), model_1.metaLearner.parameters()):
                if metaParam_1.requires_grad:
                    if metaParam_1.grad is None:
                        metaParam_1.grad = torch.zeros(localParam_1.grad.shape)
                    if metaParam_2.grad is None:
                        metaParam_2.grad = torch.zeros(localParam_2.grad.shape)
                    metaParam_1.grad += localParam_1.grad
                    metaParam_2.grad += localParam_2.grad

            model_1.zero_grad()
            model_2.zero_grad()

        # update the gradients
        optimiser_1.step()
        scheduler_1.step()
        optimiser_2.step()
        scheduler_2.step()

    def compareLines(self, lines_1, lines_2):
        for line_1, line_2 in zip(lines_1, lines_2):
            if not list(line_1.getLabels()) == list(line_2.getLabels()):
                return False
        return True

    def metaTrain(self, batch):
        accuracies = []
        losses = []
        for episode_i in range(len(batch[0])):
            data, labels = batch[0][episode_i], batch[1][episode_i]
            # split the data in support and query sets
            supportSet, supportLabels = data[0:len(data) // 2], labels[0:len(data) // 2]
            querySet, queryLabels = data[len(data) // 2:], labels[len(data) // 2:]
            # compute lines for the support set
            supportEncodings, supportLines = self.computeLines(supportSet, supportLabels)
            queryEncodings, queryLines = self.computeLines(querySet, queryLabels)
            # lines will change on query set, therefore we need to calculate them again
            # for ProtoFOMAML, since the output layer depends on the support set
            # we need to ensure that the two sets of lines are the same
            # if they are not the same, we skip this episode and carry on training
            if not self.compareLines(supportLines, queryLines):
                continue
            # for each line, carry out meta-training
            for idx, (supportLine, queryLine) in enumerate(zip(supportLines, queryLines)):
                # do not train if there is only one prototype
                if len(set(supportLine.getLabels())) == 1:
                    continue
                # perform few-shot adaptation on the support set
                innerModel_1, innerModel_2 = self.trainInnerLoop(supportLine, supportSet, supportEncodings, supportLabels)
                # calculate the loss on the query set
                self.trainOuterLoop(queryLine, innerModel_1, innerModel_2, querySet, queryEncodings, queryLabels)

    def computeLines(self, dataset, labels):
        trainingEncodings, trainingLabels = get_labelled_GLUE_episodic_training_data(dataset, labels)
        # invoke line generator and compute lines per episode
        trainingSet = {'encodings': trainingEncodings, 'labels': trainingLabels}
        lineGenerator = LineGenerator(trainingSet, PROTOTYPE_META_MODEL)
        lines = lineGenerator.generateLines(self.metaLearner)
        return trainingEncodings, lines

    def validation_step(self):
        # get the dataset and create a few shot episode
        # perform inner loop training on this model
        # evaluate on the query set
        # log the validation loss and validation accuracy
        pass

    def training_step(self, batch, batch_idx):

        # Set seeds for reproducibility
        # torch.manual_seed(42)
        # random.seed(42)

        self.metaTrain(batch)
