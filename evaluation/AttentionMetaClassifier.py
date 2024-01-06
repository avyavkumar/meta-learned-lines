import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AdamW, get_constant_schedule_with_warmup

from datautils.LEOPARDDataUtils import get_labelled_test_sentences
from datautils.LEOPARDEncoderUtils import get_model, get_tokenizer
from lines.LineGenerator import LineGenerator
from training_datasets.SentenceDataset import SentenceDataset
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils.ModelUtils import DEVICE


# TODO save the best model
# TODO implement early stopping
class AttentionMetaClassifier:

    def __init__(self, metaAttentionModel, lineGenerator: LineGenerator, trainingParams, params, labelKeys):
        self.metaAttentionModel = copy.deepcopy(metaAttentionModel)
        for name, param in self.metaAttentionModel.named_parameters():
            if 'metaLearner' in name:
                param.requires_grad = False
        self.lines = lineGenerator.generateLineIndices()
        self.trainingParams = trainingParams
        self.params = params
        self.printValidationPlot = params['printValidationPlot']
        self.printValidationLoss = params['printValidationLoss']
        self.labelKeys = labelKeys

    def createLabelDict(self, labels):
        lineLabelIndices = {}
        for i in range(len(labels)):
            lineLabelIndices[labels[i]] = i
        return lineLabelIndices

    def trainPrototypes(self, params, training_params, training_set):
        """
        get two prototypes at the end of the line
        compute the label and compare it with the true label
        generate the loss
        distribute by the reverse of the distance between the two and
        learn the parameters via backpropagation
        :param training_params: training parameters
        :param params: input parameters required for non-training purposes
        :param training_set: the training set used to train a pair of prototypes
        :return: None
        """

        # Set seeds for reproducibility
        # torch.manual_seed(42)
        # random.seed(42)

        training_sentences = training_set['sentences']
        training_encodings = training_set["encodings"]
        training_labels = training_set["labels"]

        test_validation_sentences, test_validation_labels = get_labelled_test_sentences(params["category"])
        test_validation_labels = [self.labelKeys[label] for label in test_validation_labels]

        criterion = self.getCriterion(params)

        directory = "models/" + params["encoder"] + "/cross_entropy/" + params["category"] + "/" + str(params["episode"]) + "/" + str(params["shot"]) + "/"
        Path(directory).mkdir(parents=True, exist_ok=True)

        for idx in range(len(self.lines)):
            lineLabels = self.lines[idx]
            labelDict = self.createLabelDict(lineLabels)
            # do not train if there is only one prototype, save the model and continue
            if len(set(lineLabels)) == 1:
                path = directory + "model_" + "AttentionMAML" + "_" + str(idx) + ".pt"
                torch.save({'centroids': [lineLabels[0], lineLabels[-1]],
                            'labels': lineLabels,
                            'labelDict': labelDict,
                            'model': self.metaAttentionModel}, path)
                continue

            epochs = self.getEpochs(params, lineLabels)

            # filter the dataset for the required labels
            filteredTrainingSentences, filteredTrainingLabels = self.filterSentencesByLabels(lineLabels, training_sentences, training_labels)
            filteredTrainingLabels = [labelDict[label] for label in filteredTrainingLabels]

            filteredTestValidationSentences, filteredTestValidationLabels = self.filterSentencesByLabels(lineLabels, test_validation_sentences, test_validation_labels)
            filteredTestValidationLabels = [labelDict[label] for label in filteredTestValidationLabels]

            training_dataset = SentenceDataset(filteredTrainingSentences, filteredTrainingLabels)
            train_loader = torch.utils.data.DataLoader(training_dataset, **training_params)
            test_validation_dataset = SentenceDataset(filteredTestValidationSentences, filteredTestValidationLabels)
            test_validation_loader = torch.utils.data.DataLoader(test_validation_dataset, **training_params)

            optimiser = self.getOptimiser(self.metaAttentionModel, params)
            # scheduler = self.getLearningRateScheduler(optimiser, params, len(set(filteredTrainingLabels)), len(filteredTrainingLabels))

            total_val_accuracies = []
            total_val_losses = []
            training_losses = []

            for epoch in range(epochs):  # loop over the dataset multiple times
                # switch on training mode
                self.metaAttentionModel.train()
                for i, data in enumerate(train_loader, 0):

                    # zero the parameter gradients
                    optimiser.zero_grad()

                    # get the inputs; data is a list of [inputs, labels]
                    sentences, labels = data
                    outputs = self.metaAttentionModel(sentences, labels, lineLabels[0], lineLabels[-1])

                    # # convert labels to arg values
                    # labels = self.convertLabels(line.getLabelDict(), labels)

                    # compute the loss
                    losses = criterion(outputs, labels.to(DEVICE))

                    # calculate the gradients
                    losses.mean().backward()
                    training_losses.append(losses.mean().item())

                    # update the gradients
                    optimiser.step()

                    # update the learning rate scheduler
                    # scheduler.step()

                    classes = len(set(lineLabels))

                    total_val_loss = 0.0
                    val_accuracies = []
                    with torch.no_grad():
                        # evaluate on validation set and print statistics
                        self.metaAttentionModel.eval()
                        for j, val_data in enumerate(test_validation_loader, 0):
                            val_sentences, val_labels = val_data
                            # val_labels = self.convertLabels(line.getLabelDict(), val_labels)
                            val_outputs = self.metaAttentionModel.forward_test(filteredTrainingSentences, filteredTrainingLabels, val_sentences, classes, lineLabels[0], lineLabels[-1])
                            val_losses = criterion(val_outputs, val_labels.to(DEVICE))
                            val_accuracies.append(accuracy_score(val_labels, torch.argmax(val_outputs, dim=1).detach().cpu().numpy()))
                            total_val_loss += val_losses.sum().item()
                    if self.printValidationLoss is True:
                        print("The validation loss after epoch", epoch, "and iteration", i, "is", total_val_loss, "and the accuracy is", np.mean(val_accuracies))
                    total_val_losses.append(total_val_loss)
                    total_val_accuracies.append(np.mean(val_accuracies))
                    # switch the training mode back on
                    self.metaAttentionModel.train()

            with torch.no_grad():
                self.metaAttentionModel.eval()
                path = directory + "model_" + "AttentionMAML" + "_" + str(idx) + ".pt"
                torch.save({'centroids': [lineLabels[0], lineLabels[-1]],
                            'labels': lineLabels,
                            'labelDict': labelDict,
                            'model': self.metaAttentionModel}, path)

            if self.printValidationPlot is True:
                self.generateTrainingStats(training_losses, total_val_losses, total_val_accuracies)

    def generateTrainingStats(self, training_losses, test_validation_losses, test_validation_accuracies):
        # plot the graph
        figure, axis = plt.subplots(3, 1)
        x = [i for i in range(len(test_validation_losses))]

        # losses and accuracies
        axis[0].plot(x, test_validation_losses, label='test validation losses')
        axis[0].legend(loc="upper right")
        axis[1].plot(x, training_losses, label='training losses')
        axis[1].legend(loc="upper right")
        axis[2].plot(x, test_validation_accuracies, label='test validation accuracies')
        axis[2].legend(loc="upper right")
        plt.legend()
        plt.show()

        # show the index used for obtaining the lowest test validation loss
        min_index = np.argmin(test_validation_losses)
        print("Lowest test validation loss is", test_validation_losses[min_index], "at index", min_index)
        print("The corresponding test validation accuracy is", test_validation_accuracies[min_index])
        print("The accuracy at the last iteration is", test_validation_accuracies[-1], "\n")

    def filterEncodingsByLabels(self, labels, training_data, training_labels):
        filteredTrainingData = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingData.append(training_data[i].detach().cpu().numpy())
                filteredTrainingLabels.append(training_labels[i])
        return torch.Tensor(np.array(filteredTrainingData)), np.array(filteredTrainingLabels)

    def filterSentencesByLabels(self, labels, training_data, training_labels):
        filteredTrainingSentences = []
        filteredTrainingLabels = []
        for i in range(len(training_labels)):
            if training_labels[i] in labels:
                filteredTrainingSentences.append(training_data[i])
                filteredTrainingLabels.append(training_labels[i])
        return filteredTrainingSentences, np.array(filteredTrainingLabels)

    def getPredictions(self, outputs, line):
        predictedLabels = []
        argmaxLabels = torch.argmax(outputs, dim=1).tolist()
        reverseLookup = {}
        for key, value in line.getLabelDict().items():
            reverseLookup[value] = key
        for i in range(len(argmaxLabels)):
            predictedLabels.append(reverseLookup[argmaxLabels[i]])
        return predictedLabels

    def convertLabels(self, labelDict, labels):
        argLabels = []
        for i in range(len(labels)):
            argLabels.append(labelDict[labels[i].item()])
        return torch.Tensor(argLabels).long()

    def getCriterion(self, params):
        return nn.CrossEntropyLoss(reduction=params['reduction'])

    def getOptimiser(self, model, params):
        return AdamW(model.parameters(), lr=params['learning_rate'][params['shot']])

    def getLearningRateScheduler(self, optimiser, params, classes, dataPoints):
        totalSteps = params['epochs'][params['shot']][classes] * math.ceil(dataPoints / params['batch_size'][params['shot']])
        return CosineAnnealingLR(optimizer=optimiser, T_max=totalSteps, eta_min=1e-5, verbose=True)

    def getEpochs(self, params, labels):
        return params['epochs'][params['shot']][len(set(labels))]

    def getLines(self):
        return self.lines
