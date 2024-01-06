import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup

from datautils.LEOPARDDataUtils import get_labelled_test_sentences
from datautils.LEOPARDEncoderUtils import get_model, get_tokenizer
from lines.LineGenerator import LineGenerator
from prototypes.models.PrototypeMetaLinearModel import PrototypeMetaLinearUnifiedModel
from training_datasets.SentenceDataset import SentenceDataset
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils.ModelUtils import DEVICE, CPU_DEVICE


# TODO save the best model
# TODO implement early stopping
class MetaClassifier:

    def __init__(self, metaLearner, lineGenerator: LineGenerator, trainingParams, params, labelKeys):
        self.metaLearner = copy.deepcopy(metaLearner)
        self.lines = lineGenerator.generateLines()
        self.trainingParams = trainingParams
        self.params = params
        self.printValidationPlot = params['printValidationPlot']
        self.printValidationLoss = params['printValidationLoss']
        self.labelKeys = labelKeys

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
        training_labels = training_set["labels"]

        test_validation_sentences, test_validation_labels = get_labelled_test_sentences(params["category"])
        test_validation_labels = [self.labelKeys[label] for label in test_validation_labels]

        criterion = self.getCriterion(params)

        directory = "models/" + params["encoder"] + "/cross_entropy/" + params["category"] + "/" + str(params["episode"]) + "/" + str(params["shot"]) + "/"
        Path(directory).mkdir(parents=True, exist_ok=True)
        accuracy = 0.0

        for idx, line in enumerate(self.lines):
            # do not train if there is only one prototype, save the model and continue
            if len(set(line.getLabels())) == 1:
                print("Single point line generated...")
                path = directory + "model_" + line.getModelType() + "_" + str(idx) + ".pt"
                torch.save({'centroids': line.getCentroids(),
                            'labels': line.getLabels(),
                            'modelType': line.getModelType(),
                            'labelDict': line.getLabelDict(),
                            'model': PrototypeMetaLinearUnifiedModel(copy.deepcopy(self.metaLearner), classes=1).state_dict()}, path)
                continue

            epochs = self.getEpochs(params, len(line.getLabels()))

            # filter the dataset for the required labels
            filteredTrainingSentences, filteredTrainingLabels = self.filterSentencesByLabels(line.getLabels(), training_sentences, training_labels)
            filteredTrainingLabels = [line.getLabelDict()[label] for label in filteredTrainingLabels]

            filteredTestValidationSentences, filteredTestValidationLabels = self.filterSentencesByLabels(line.getLabels(), test_validation_sentences, test_validation_labels)
            filteredTestValidationLabels = [line.getLabelDict()[label] for label in filteredTestValidationLabels]

            training_dataset = SentenceDataset(filteredTrainingSentences, filteredTrainingLabels)
            train_loader = torch.utils.data.DataLoader(training_dataset, **training_params)
            test_validation_dataset = SentenceDataset(filteredTestValidationSentences, filteredTestValidationLabels)
            test_validation_loader = torch.utils.data.DataLoader(test_validation_dataset, **training_params)
            model = PrototypeMetaLinearUnifiedModel(copy.deepcopy(self.metaLearner), classes=len(set(filteredTrainingLabels)))
            model.to(DEVICE)

            optimiser = self.getOptimiser(model, params)
            scheduler = self.getLearningRateScheduler(optimiser, params, classes=len(set(filteredTrainingLabels)))

            total_val_accuracies = []
            total_val_losses = []
            training_losses = []

            for epoch in range(epochs):  # loop over the dataset multiple times
                # switch on training mode
                model.train()
                for i, data in enumerate(train_loader, 0):
                    # zero the parameter gradients
                    optimiser.zero_grad()

                    # get the inputs; data is a list of [inputs, labels]
                    sentences, labels = data
                    label_1 = line.getLabelDict()[line.getLabels()[0]]
                    label_2 = line.getLabelDict()[line.getLabels()[-1]]

                    outputs, distances_1, distances_2 = model(sentences, labels, label_1, label_2)

                    # compute the loss
                    losses = criterion(outputs, labels.to(DEVICE))
                    # calculate the gradients
                    losses.mean().backward()
                    training_losses.append(losses.mean().item())

                    # multiply the calculated gradients of each model by a scaling factor
                    self.updateGradients(losses, model, distances_1, distances_2)

                    # update the gradients
                    optimiser.step()
                    scheduler.step()

                    if epoch == epochs - 1:
                        total_val_loss = 0.0
                        val_accuracies = []
                        val_predictions = []
                        all_val_labels = []
                        with torch.no_grad():
                            # evaluate on validation set and print statistics
                            model.eval()
                            for j, val_data in enumerate(test_validation_loader, 0):
                                val_sentences, val_labels = val_data
                                # val_labels = self.convertLabels(line.getLabelDict(), val_labels)
                                val_outputs = model.forward_test(sentences, labels, val_sentences, label_1, label_2)
                                val_losses = criterion(val_outputs, val_labels.to(DEVICE))
                                val_predictions.extend(torch.argmax(val_outputs, dim=1).detach().cpu().numpy())
                                all_val_labels.extend(val_labels)
                                val_accuracies.append(accuracy_score(val_labels, torch.argmax(val_outputs, dim=1).detach().cpu().numpy()))
                                total_val_loss += val_losses.sum().item()
                        accuracy = accuracy_score(all_val_labels, val_predictions)
                        if self.printValidationLoss is True:
                            print("The validation loss after epoch", epoch, "and iteration", i, "is", total_val_loss, "and the accuracy is", accuracy_score(all_val_labels, val_predictions))
                        total_val_losses.append(total_val_loss)
                        total_val_accuracies.append(np.mean(val_accuracies))
                        # switch the training mode back on
                        model.train()

            with torch.no_grad():
                model.eval()
                path = directory + "model_" + line.getModelType() + "_" + str(idx) + ".pt"
                # calculate the centroids
                centroids = []
                # for label in line.getLabels():
                #     centroids.append(torch.mean(model.metaLearner([filteredTrainingSentences[x] for x in range(len(filteredTrainingSentences)) if filteredTrainingLabels[x] == line.getLabelDict()[label]]), dim=0))
                torch.save({
                    'centroids': line.getCentroids(),
                    'labels': line.getLabels(),
                    'modelType': line.getModelType(),
                    'labelDict': line.getLabelDict(),
                    'model': model.state_dict()}, path)

            if self.printValidationPlot is True:
                self.generateTrainingStats(training_losses, total_val_losses, total_val_accuracies)
        return accuracy

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

    def updateGradients(self, losses, model, distances_1, distances_2):
        losses_1 = losses.clone().detach().cpu()
        losses_2 = losses.clone().detach().cpu()
        losses_1 = distances_2.to(CPU_DEVICE).squeeze(1) / torch.sum(torch.cat((distances_1.to(CPU_DEVICE), distances_2.to(CPU_DEVICE)), 1), dim=1) * losses_1
        losses_2 = distances_1.to(CPU_DEVICE).squeeze(1) / torch.sum(torch.cat((distances_1.to(CPU_DEVICE), distances_2.to(CPU_DEVICE)), 1), dim=1) * losses_2
        loss_ratio_1 = losses_1.sum() / (losses_1.sum() + losses_2.sum())
        loss_ratio_2 = losses_2.sum() / (losses_1.sum() + losses_2.sum())
        model.scaleModelGradients(loss_ratio_1, loss_ratio_2)

    def convertLabels(self, labelDict, labels):
        argLabels = []
        for i in range(len(labels)):
            argLabels.append(labelDict[labels[i].item()])
        return torch.Tensor(argLabels).long()

    def getCriterion(self, params):
        return nn.CrossEntropyLoss(reduction=params['reduction'])

    def getOptimiser(self, model, params):
        return SGD([{'params': model.metaLearner.parameters(), 'lr': params['inner_learning_rate'][params['shot']]},
                    {'params': model.linear_1.parameters(), 'lr': params['output_learning_rate'][params['shot']]},
                    {'params': model.linear_2.parameters(), 'lr': params['output_learning_rate'][params['shot']]}], momentum=0.9, nesterov=True)
        # return SGD([{'params': model.metaLearner.parameters(), 'lr': params['inner_learning_rate'][params['shot']]},
        #             {'params': model.linear_1.parameters(), 'lr': params['output_learning_rate'][params['shot']]},
        #             {'params': model.linear_2.parameters(), 'lr': params['output_learning_rate'][params['shot']]}])

    def getLearningRateScheduler(self, optimiser, params, classes):
        return CosineAnnealingLR(optimiser,  T_max=self.getEpochs(params, classes), eta_min=params['min_lr'][params['shot']])

    def computeLabelsAndDistances(self, sentences, encodings, prototype_1, prototype_2):

        output_1 = prototype_1.getPrototypeModel()(sentences)
        output_2 = prototype_2.getPrototypeModel()(sentences)

        # get distances from the prototypes for all inputs
        distances_1 = []
        distances_2 = []
        for i in range(encodings.shape[0]):
            distances_1.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_1.getLocation().detach().cpu().numpy()))
            distances_2.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_2.getLocation().detach().cpu().numpy()))

        distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1).to(DEVICE)
        distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1).to(DEVICE)

        # compute the weighted probability distribution
        outputs = output_1 / distances_1 + output_2 / distances_2

        # return the final weighted probability distribution
        return outputs, distances_1, distances_2

    def getEpochs(self, params, classes):
        return params['epochs'][params['shot']][classes]

    def getLines(self):
        return self.lines
