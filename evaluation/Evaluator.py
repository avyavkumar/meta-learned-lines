import copy

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from os import listdir
from os.path import isfile, join

from datautils.LEOPARDDataUtils import get_labelled_test_sentences
from lines.Line import Line

from lines.lo_shot_utils import dist_to_line_multiD
from training_datasets.SentenceEncodingDataset import SentenceEncodingDataset
from utils.ModelUtils import CPU_DEVICE, DEVICE


class Evaluator:
    def __init__(self):
        self.test_sentences = None
        self.test_encodings = None
        self.test_labels = None
        self.evaluated_test_data = False

    def evaluate(self, metaLearner, label_keys, test_params, category, shot, episode):
        classes = len(label_keys.keys())
        predictions = []
        true_labels = []
        assignments = []
        # construct Line objects from the saved models
        lines = self.getLines(metaLearner, test_params, category, shot, episode)

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
            required_test_encodings = np.array([self.test_encodings[x].detach().cpu().numpy() for x in range((len(assignments))) if assignments[x] == i])
            required_test_sentences = [self.test_sentences[x] for x in range((len(assignments))) if assignments[x] == i]
            required_test_encodings = torch.from_numpy(required_test_encodings)
            if required_test_encodings.shape[0] > 0:
                required_test_labels = [self.test_labels[x] for x in range((len(assignments))) if assignments[x] == i]
                test_dataset = SentenceEncodingDataset(required_test_sentences, required_test_encodings, required_test_labels)
                params = {'batch_size': 32}
                test_loader = torch.utils.data.DataLoader(test_dataset, **params)
                for point_i, data in enumerate(test_loader):
                    sentences, encodings, labels = data
                    outputs = self.computeLabels(sentences, encodings, lines[i])
                    # map predictions to labels
                    predictions_i = self.getPredictions(outputs, lines[i])
                    predictions.extend(predictions_i)
                    true_labels.extend(labels)

        print("For category", category, "and shot =", str(shot) + "...")
        print("Lines used are", len(lines))
        print("Number of classifications are", classes)
        print("Macro f1 score is", f1_score(true_labels, predictions, average='macro'))
        print("Accuracy is", accuracy_score(true_labels, predictions))
        print("Correctly classified points are", np.sum(np.array(true_labels) == np.array(predictions)), "/", len(true_labels), "\n")

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

    def getLines(self, metaLearner, test_params, category, shot, episode):
        path = "models/" + str(test_params['encoder']) + "/" + str(test_params['type']) + "/" + category + "/" + str(episode) + "/" + str(shot) + "/"
        model_paths = [f for f in listdir(path) if isfile(join(path, f))]
        lines = []
        for model_path in model_paths:
            modelCheckpoint = torch.load(path + model_path)
            centroids = modelCheckpoint['centroids']
            labels = modelCheckpoint['labels']
            modelType = modelCheckpoint['modelType']
            labelDict = modelCheckpoint['labelDict']
            line = Line(totalClasses=len(set(labels)), centroids=centroids, labels=labels, modelType=modelType, metaLearner=metaLearner, labelDict=labelDict)
            line.getFirstPrototype().getPrototypeModel().load_state_dict(modelCheckpoint['prototype_1'])
            line.getFirstPrototype().getPrototypeModel().eval()
            line.getSecondPrototype().getPrototypeModel().load_state_dict(modelCheckpoint['prototype_2'])
            line.getSecondPrototype().getPrototypeModel().eval()
            lines.append(line)
        return lines

    def getTestEncodings(self, metaLearner, data, labels):
        with torch.no_grad():
            test_encodings = []
            test_labels = labels
            for i in range(len(test_labels)):
                encoding = metaLearner(data[i]).cpu().detach().reshape(-1)
                test_encodings.append(encoding)
                del encoding
            return torch.stack(test_encodings, dim=0), torch.Tensor(test_labels)

    def computeLabels(self, sentences, encodings, line):
        with torch.no_grad():
            prototype_1 = line.getFirstPrototype()
            prototype_2 = line.getSecondPrototype()
            prototype_1.getPrototypeModel().to(DEVICE)
            prototype_2.getPrototypeModel().to(DEVICE)
            output_1 = prototype_1.getPrototypeModel()(sentences).to(CPU_DEVICE)
            output_2 = prototype_2.getPrototypeModel()(sentences).to(CPU_DEVICE)
            # get distances from the prototypes for all inputs
            distances_1 = []
            distances_2 = []
            for i in range(encodings.shape[0]):
                distances_1.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_1.getLocation().detach().cpu().numpy()))
                distances_2.append(np.linalg.norm(encodings[i].detach().cpu().numpy() - prototype_2.getLocation().detach().cpu().numpy()))
            distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1)
            distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1)
            # compute the weighted probability distribution
            outputs = output_1 / distances_1 + output_2 / distances_2
            # delete the outputs
            del output_1, output_2
            # return the final weighted probability distribution
            return outputs
