import copy

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from os import listdir
from os.path import isfile, join

from datautils.LEOPARDDataUtils import get_labelled_test_sentences
from lines.Line import Line

from lines.lo_shot_utils import dist_to_line_multiD
from prototypes.models.PrototypeMetaLinearModel import PrototypeMetaLinearUnifiedModel
from prototypes.models.PrototypeMetaModel import PrototypeMetaModel
from training_datasets.SentenceDataset import SentenceDataset
from utils.ModelUtils import CPU_DEVICE, DEVICE


class Evaluator:
    def __init__(self):
        self.test_sentences = None
        self.test_encodings = None
        self.test_labels = None
        self.evaluated_test_data = False
        self.models = None

    def evaluate(self, metaLearner, supportSet, supportLabels, label_keys, test_params, category, shot, episode):
        classes = len(label_keys.keys())
        predictions = []
        true_labels = []
        assignments = []
        # construct Line objects from the saved models
        lines, models = self.getLinesAndModels(test_params, category, shot, episode)

        if not self.evaluated_test_data:
            self.test_sentences, categorical_test_labels = get_labelled_test_sentences(category)
            self.test_labels = [label_keys[label] for label in categorical_test_labels]
            self.test_encodings, self.test_labels = self.getTestEncodings(metaLearner, self.test_sentences, self.test_labels)
            self.evaluated_test_data = True

        for point in self.test_encodings:
            dists = [dist_to_line_multiD(point.detach().cpu().numpy(), line.getFirstPrototype().getLocation().detach().cpu().numpy(), line.getSecondPrototype().getLocation().detach().cpu().numpy()) for line in lines]
            nearest = np.argmin(dists)
            assignments.append(nearest)
        for i in range(len(lines)):
            required_test_sentences = [self.test_sentences[x] for x in range((len(assignments))) if assignments[x] == i]
            # fetch the required support set and support labels as well
            filteredTrainingSentences, filteredTrainingLabels = self.filterSentencesByLabels(lines[i].getLabels(), supportSet, supportLabels)
            filteredTrainingLabels = [lines[i].getLabelDict()[label] for label in filteredTrainingLabels]
            if len(required_test_sentences) > 0:
                required_test_labels = [self.test_labels[x] for x in range((len(assignments))) if assignments[x] == i]
                test_dataset = SentenceDataset(required_test_sentences, required_test_labels)
                params = {'batch_size': 64}
                test_loader = torch.utils.data.DataLoader(test_dataset, **params)
                for point_i, data in enumerate(test_loader):
                    test_sentences, test_labels = data
                    models[i].to(DEVICE)
                    outputs = models[i].forward_test(filteredTrainingSentences, filteredTrainingLabels, test_sentences, lines[i].getLabelDict()[lines[i].getLabels()[0]], lines[i].getLabelDict()[lines[i].getLabels()[-1]])
                    # map predictions to labels
                    predictions_i = self.getPredictions(outputs, lines[i])
                    predictions.extend(predictions_i)
                    true_labels.extend(test_labels)
            models[i].to(CPU_DEVICE)

        print("For category", category, "and shot =", str(shot) + "...")
        print("Lines used are", len(lines))
        print("Number of classifications are", classes)
        print("Macro f1 score is", f1_score(true_labels, predictions, average='macro'))
        print("Accuracy is", accuracy_score(true_labels, predictions))
        print("Correctly classified points are", np.sum(np.array(true_labels) == np.array(predictions)), "/", len(true_labels), "\n")

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
        # if only one label exists, add a special entry at the 0th index to refer to this
        if len(set(line.getLabels())) == 1:
            reverseLookup[0] = line.getLabels()[0]
        for i in range(len(argmaxLabels)):
            predictedLabels.append(reverseLookup[argmaxLabels[i]])
        return predictedLabels

    def getLinesAndModels(self, test_params, category, shot, episode):
        path = "models/" + str(test_params['encoder']) + "/" + str(test_params['type']) + "/" + category + "/" + str(episode) + "/" + str(shot) + "/"
        model_paths = [f for f in listdir(path) if isfile(join(path, f))]
        lines = []
        models = []
        for model_path in model_paths:
            modelCheckpoint = torch.load(path + model_path)
            centroids = modelCheckpoint['centroids']
            labels = modelCheckpoint['labels']
            modelType = modelCheckpoint['modelType']
            labelDict = modelCheckpoint['labelDict']
            line = Line(totalClasses=len(set(labels)), centroids=centroids, labels=labels, modelType=modelType, labelDict=labelDict)
            model = PrototypeMetaLinearUnifiedModel(PrototypeMetaModel(), len(set(labels)))
            model.load_state_dict(modelCheckpoint['model'])
            models.append(model)
            lines.append(line)
        return lines, models

    def getTestEncodings(self, metaLearner, data, labels):
        with torch.no_grad():
            metaLearner.eval()
            test_encodings = []
            test_labels = labels
            for i in range(len(test_labels)):
                encoding = metaLearner(data[i]).cpu().detach().reshape(-1)
                test_encodings.append(encoding)
                del encoding
            return torch.stack(test_encodings, dim=0), torch.Tensor(test_labels)
