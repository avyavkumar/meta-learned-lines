import torch
import numpy as np
import torch.nn as nn

from learn2learn.algorithms import LightningMAML
from learn2learn.utils.lightning import EpisodicBatcher
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_constant_schedule_with_warmup
from pathlib import Path

import torch.nn.functional as F
from datautils.GLUEEncoderUtils import get_labelled_GLUE_episodic_training_data
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from utils.Constants import FOMAML, HIDDEN_MODEL_SIZE, PROTOTYPE_META_MODEL

from datautils.GLUEDataUtils import get_random_GLUE_dataset
from lines.LineGenerator import LineGenerator
from training_datasets.EncodingDataset import EncodingDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from datautils.LEOPARDEncoderUtils import read_test_data as read_test_data_bert
from training_datasets.GLUEMetaDataset import GLUEMetaDataset
import pytorch_lightning as L


# TODO save the best model
# TODO implement early stopping#
# TODO check if the training is happening as expected - meta-layer is consistent across all layers
class FOMAML(L.LightningModule):

    def __init__(self, outerLR, innerLR, outputLR, steps):
        super().__init__()
        self.save_hyperparameters()
        # self.printValidationPlot = params['printValidationPlot']
        # self.printValidationLoss = params['printValidationLoss']
        self.metaLearner = PrototypeMetaModel()

    def trainInnerLoop(self):
        pass

    def trainOuterLoop(self, batch):
        pass

    def computeLines(self, dataset: GLUEMetaDataset, params):
        """
        we have got the lines and the meta + linear model is attached at the ends of the line
        write a loss function which works off cross entropy but splits the total loss into the fraction of distances
          it is capable of calculating accuracy etc
        we need to start training
          get the hyperparameters and meta-hyperparameters
              inner loop learning rate, outer loop learning rate, dropout, adaptation steps
              learning rates are different for linear layer and meta-model - modify the code in lightning_maml
          initialise two trainers
          pick an episode and compute the loss using the custom loss function for both the networks
          attach those losses to the trainers and ensure that
              (a) linear layers are updated according to the loss fraction assigned to them
              (b) meta-layers are updated twice (essentially boils down to cross entropy loss)
          check on the validation set
          save the best meta-model - check if there is a callback by default
        """

        # Set seeds for reproducibility
        # torch.manual_seed(42)
        # random.seed(42)

        # sample episodes in the batch
        episodes = []
        for _ in range(params['batch_size']):
            episodes.append(dataset.getTask())

        linesPerEpisode = []
        # get training encodings and centroids for all episodes
        for episode_i in range(len(episodes)):
            trainingEncodings, trainingLabels = get_labelled_GLUE_episodic_training_data(episodes[episode_i])
            # invoke line generator and compute lines per episode
            trainingSet = {'encodings': trainingEncodings, 'labels': trainingLabels}
            lineGenerator = LineGenerator(trainingSet, PROTOTYPE_META_MODEL)
            # store these lines in a list
            linesPerEpisode.append(lineGenerator.generateLines(self.metaLearner))

        return linesPerEpisode

    def training_step(self, batch, batch_idx):
        self.trainOuterLoop(batch)




    #     encodings = training_set["encodings"]
    #     training_labels = training_set["labels"]
    #
    #     test_validation_data = []
    #     test_validation_labels = []
    #
    #     if params['encoder'] == "bert":
    #         test_validation_data, test_validation_labels = read_test_data_bert(params["category"])
    #         test_validation_labels = [self.labelKeys[label] for label in test_validation_labels]
    #
    #     criterion = self.getCriterion(params)
    #
    #     directory = "models/" + params["encoder"] + "/cross_entropy/" + params["category"] + "/" + str(
    #         params["episode"]) + "/" + str(params["shot"]) + "/"
    #     Path(directory).mkdir(parents=True, exist_ok=True)
    #
    #     for idx, line in enumerate(self.lines):
    #         # do not train if there is only one prototype, save the model and continue
    #         if len(set(line.getLabels())) == 1:
    #             path = directory + "model_" + line.getModelType() + "_" + str(idx) + ".pt"
    #             torch.save({
    #                 'centroids': line.getCentroids(),
    #                 'labels': line.getLabels(),
    #                 'modelType': line.getModelType(),
    #                 'labelDict': line.getLabelDict(),
    #                 'prototype_1': line.getFirstPrototype().getPrototypeModel().state_dict(),
    #                 'prototype_2': line.getSecondPrototype().getPrototypeModel().state_dict(),
    #             }, path)
    #             continue
    #
    #         epochs = self.getEpochs(params, len(line.getLabels()))
    #
    #         # filter the datautils for the required labels
    #         filteredTrainingData, filteredTrainingLabels = self.filterEncodingsByLabels(line.getLabels(), encodings,
    #                                                                                     training_labels)
    #         filteredTrainingLabels = [line.getLabelDict()[label] for label in filteredTrainingLabels]
    #         filteredTestValidationData, filteredTestValidationLabels = self.filterEncodingsByLabels(line.getLabels(),
    #                                                                                                 test_validation_data,
    #                                                                                                 test_validation_labels)
    #         filteredTestValidationLabels = [line.getLabelDict()[label] for label in filteredTestValidationLabels]
    #
    #         training_dataset = EncodingDataset(filteredTrainingData, filteredTrainingLabels)
    #         train_loader = torch.utils.data.DataLoader(training_dataset, **training_params)
    #         test_validation_dataset = EncodingDataset(filteredTestValidationData, filteredTestValidationLabels)
    #         test_validation_loader = torch.utils.data.DataLoader(test_validation_dataset, **training_params)
    #
    #         prototype_1 = line.getFirstPrototype()
    #         prototype_2 = line.getSecondPrototype()
    #         optimiser_1 = self.getOptimiser(prototype_1.getPrototypeModel(), params)
    #         optimiser_2 = self.getOptimiser(prototype_2.getPrototypeModel(), params)
    #         scheduler_1 = self.getLearningRateScheduler(optimiser_1, params)
    #         scheduler_2 = self.getLearningRateScheduler(optimiser_2, params)
    #
    #         total_val_accuracies = []
    #         total_val_losses = []
    #         training_losses = []
    #
    #         for epoch in range(epochs):  # loop over the datautils multiple times
    #             # switch on training mode
    #             prototype_1.getPrototypeModel().train()
    #             prototype_2.getPrototypeModel().train()
    #             for i, data in enumerate(train_loader, 0):
    #
    #                 # zero the parameter gradients
    #                 optimiser_1.zero_grad()
    #                 optimiser_2.zero_grad()
    #
    #                 # get the inputs; data is a list of [inputs, labels]
    #                 inputs, labels = data
    #                 outputs, distances_1, distances_2 = self.computeLabelsAndDistances(inputs, prototype_1, prototype_2)
    #
    #                 # # convert labels to arg values
    #                 # labels = self.convertLabels(line.getLabelDict(), labels)
    #
    #                 # compute the loss
    #                 losses = criterion(outputs, labels)
    #
    #                 # calculate the gradients
    #                 losses.sum().backward()
    #
    #                 training_losses.append(losses.sum().item())
    #
    #                 # multiply the calculated gradients of each model by a scaling factor
    #                 self.updateGradients(losses, prototype_1, prototype_2, distances_1, distances_2)
    #
    #                 # update the gradients
    #                 optimiser_1.step()
    #                 optimiser_2.step()
    #
    #                 # update the learning rate scheduler
    #                 scheduler_1.step()
    #                 scheduler_2.step()
    #
    #                 # get the accuracy
    #                 accuracy = accuracy_score(labels, torch.argmax(outputs, dim=1))
    #
    #                 total_val_loss = 0.0
    #                 val_accuracies = []
    #                 with torch.no_grad():
    #                     # evaluate on validation set and print statistics
    #                     prototype_1.getPrototypeModel().eval()
    #                     prototype_2.getPrototypeModel().eval()
    #                     for j, val_data in enumerate(test_validation_loader, 0):
    #                         val_inputs, val_labels = val_data
    #                         # val_labels = self.convertLabels(line.getLabelDict(), val_labels)
    #                         val_outputs, _, _ = self.computeLabelsAndDistances(val_inputs, prototype_1, prototype_2)
    #                         val_losses = criterion(val_outputs, val_labels)
    #                         val_accuracies.append(accuracy_score(val_labels, torch.argmax(val_outputs, dim=1)))
    #                         total_val_loss += val_losses.sum().item()
    #                 if self.printValidationLoss is True:
    #                     print("The validation loss after epoch", epoch, "and iteration", i, "is", total_val_loss,
    #                           "and the accuracy is", np.mean(val_accuracies))
    #                 total_val_losses.append(total_val_loss)
    #                 total_val_accuracies.append(np.mean(val_accuracies))
    #                 # switch the training mode back on
    #                 prototype_1.getPrototypeModel().train()
    #                 prototype_2.getPrototypeModel().train()
    #
    #         with torch.no_grad():
    #             prototype_1.getPrototypeModel().eval()
    #             prototype_2.getPrototypeModel().eval()
    #             path = directory + "model_" + line.getModelType() + "_" + str(idx) + ".pt"
    #             torch.save({
    #                 'centroids': line.getCentroids(),
    #                 'labels': line.getLabels(),
    #                 'modelType': line.getModelType(),
    #                 'labelDict': line.getLabelDict(),
    #                 'prototype_1': prototype_1.getPrototypeModel().state_dict(),
    #                 'prototype_2': prototype_2.getPrototypeModel().state_dict()
    #             }, path)
    #
    #         if self.printValidationPlot is True:
    #             self.generateTrainingStats(training_losses, total_val_losses, total_val_accuracies)
    #
    # def generateTrainingStats(self, training_losses, test_validation_losses, test_validation_accuracies):
    #     # plot the graph
    #     figure, axis = plt.subplots(3, 1)
    #     x = [i for i in range(len(test_validation_losses))]
    #
    #     # losses and accuracies
    #     axis[0].plot(x, test_validation_losses, label='test validation losses')
    #     axis[0].legend(loc="upper right")
    #     axis[1].plot(x, training_losses, label='training losses')
    #     axis[1].legend(loc="upper right")
    #     axis[2].plot(x, test_validation_accuracies, label='test validation accuracies')
    #     axis[2].legend(loc="upper right")
    #     plt.legend()
    #     plt.show()
    #
    #     # show the index used for obtaining the lowest test validation loss
    #     min_index = np.argmin(test_validation_losses)
    #     print("Lowest test validation loss is", test_validation_losses[min_index], "at index", min_index)
    #     print("The corresponding test validation accuracy is", test_validation_accuracies[min_index])
    #     print("The accuracy at the last iteration is", test_validation_accuracies[-1], "\n")
    #
    # def filterEncodingsByLabels(self, labels, training_data, training_labels):
    #     filteredTrainingData = []
    #     filteredTrainingLabels = []
    #     for i in range(len(training_labels)):
    #         if training_labels[i] in labels:
    #             filteredTrainingData.append(training_data[i].detach().numpy())
    #             filteredTrainingLabels.append(training_labels[i])
    #     return torch.Tensor(np.array(filteredTrainingData)), np.array(filteredTrainingLabels)
    #
    # def getPredictions(self, outputs, line):
    #     predictedLabels = []
    #     argmaxLabels = torch.argmax(outputs, dim=1).tolist()
    #     reverseLookup = {}
    #     for key, value in line.getLabelDict().items():
    #         reverseLookup[value] = key
    #     for i in range(len(argmaxLabels)):
    #         predictedLabels.append(reverseLookup[argmaxLabels[i]])
    #     return predictedLabels
    #
    # def updateGradients(self, losses, prototype_1, prototype_2, distances_1, distances_2):
    #     """
    #     The pseudocode for scaling gradients is as follows
    #         for each loss
    #           divide the loss by the sum of the distances
    #           compute loss for the first model by multiplying loss*(distance_2/(distance_1 + distance_2))
    #           compute loss for the second model by multiplying loss*(distance_2/(distance_1 + distance_2))
    #         get the ratio of the losses
    #         multiply the gradients by the factor required
    #     :param losses: the losses per mini-batch
    #     :param prototype_1: the first soft-label prototype in the line
    #     :param prototype_2: the second soft-label prototype in the line
    #     :param distances_1: the distances from the first soft-label prototype in the mini-batch
    #     :param distances_2: the distances from the second soft-label prototype in the mini-batch
    #     """
    #     losses_1 = losses.clone().detach()
    #     losses_2 = losses.clone().detach()
    #
    #     losses_1 = distances_2.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_1
    #     losses_2 = distances_1.squeeze(1) / torch.sum(torch.cat((distances_1, distances_2), 1), dim=1) * losses_2
    #
    #     loss_ratio_1 = (losses_1.sum() / (losses_1.sum() + losses_2.sum()))
    #     loss_ratio_2 = (losses_2.sum() / (losses_1.sum() + losses_2.sum()))
    #
    #     prototype_1.getPrototypeModel().scaleGradients(loss_ratio_1)
    #     prototype_2.getPrototypeModel().scaleGradients(loss_ratio_2)
    #
    # def convertLabels(self, labelDict, labels):
    #     argLabels = []
    #     for i in range(len(labels)):
    #         argLabels.append(labelDict[labels[i].item()])
    #     return torch.Tensor(argLabels).long()
    #
    # def getCriterion(self, params):
    #     return nn.CrossEntropyLoss(reduction=params['reduction'])
    #
    # def getOptimiser(self, model, params):
    #     return AdamW(model.parameters(), lr=params['learning_rate'][params['shot']])
    #
    # def getLearningRateScheduler(self, optimiser, params):
    #     return get_constant_schedule_with_warmup(optimiser, num_warmup_steps=params['warmupSteps'])
    #
    # def computeLabelsAndDistances(self, inputs, prototype_1, prototype_2):
    #
    #     output_1 = prototype_1.getPrototypeModel()(inputs)
    #     output_2 = prototype_2.getPrototypeModel()(inputs)
    #
    #     # get distances from the prototypes for all inputs
    #     distances_1 = []
    #     distances_2 = []
    #     for i in range(inputs.shape[0]):
    #         distances_1.append(np.linalg.norm(inputs[i].detach().numpy() - prototype_1.getLocation().detach().numpy()))
    #         distances_2.append(np.linalg.norm(inputs[i].detach().numpy() - prototype_2.getLocation().detach().numpy()))
    #
    #     distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1)
    #     distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1)
    #
    #     # compute the weighted probability distribution
    #     outputs = output_1 / distances_1 + output_2 / distances_2
    #
    #     # return the final weighted probability distribution
    #     return outputs, distances_1, distances_2
    #
    # def getEpochs(self, params, classes):
    #     return params['epochs'][params['shot']][classes]
    #
    # def getLines(self):
    #     return self.lines
