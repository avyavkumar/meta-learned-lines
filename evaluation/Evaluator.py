import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from os import listdir
from os.path import isfile, join

from lines.Line import Line

from datautils.LEOPARDEncoderUtils import read_test_data
from lines.lo_shot_utils import dist_to_line_multiD


class Evaluator:
    def __init__(self, label_keys, test_params):
        self.label_keys = label_keys
        self.test_params = test_params

    def evaluate(self, category, shot, episode):
        classes = len(self.label_keys.keys())
        predictions = []
        true_labels = []
        assignments = []
        # construct Line objects from the saved models
        lines = self.getLines(category, shot, episode)

        test_encodings, categorical_test_labels = read_test_data(category)
        test_labels = [self.label_keys[label] for label in categorical_test_labels]

        for point in test_encodings:
            dists = [dist_to_line_multiD(point.detach().numpy(), line.getFirstPrototype().getLocation().detach().numpy(),
                                         line.getSecondPrototype().getLocation().detach().numpy()) for line in lines]
            nearest = np.argmin(dists)
            assignments.append(nearest)
        for i in range(len(lines)):
            required_test_encodings = np.array([test_encodings[x].detach().numpy() for x in range((len(assignments))) if assignments[x] == i])
            required_test_encodings = torch.from_numpy(required_test_encodings)
            if required_test_encodings.shape[0] > 0:
                required_test_labels = [test_labels[x] for x in range((len(assignments))) if assignments[x] == i]
                outputs = self.computeLabels(required_test_encodings, lines[i])
                # map predictions to labels
                predictions_i = self.getPredictions(outputs, lines[i])
                predictions.extend(predictions_i)
                true_labels.extend(required_test_labels)

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

    def getLines(self, category, shot, episode):
        path = "models/" + str(self.test_params['encoder']) + "/" + str(self.test_params['type']) + "/" + category + "/" + str(episode) + "/" + str(shot) + "/"
        model_paths = [f for f in listdir(path) if isfile(join(path, f))]
        lines = []
        for model_path in model_paths:
            modelCheckpoint = torch.load(path + model_path)
            centroids = modelCheckpoint['centroids']
            labels = modelCheckpoint['labels']
            modelType = modelCheckpoint['modelType']
            labelDict = modelCheckpoint['labelDict']
            line = Line(centroids, labels, modelType, labelDict)
            line.getFirstPrototype().getPrototypeModel().load_state_dict(modelCheckpoint['prototype_1'])
            line.getFirstPrototype().getPrototypeModel().eval()
            line.getSecondPrototype().getPrototypeModel().load_state_dict(modelCheckpoint['prototype_2'])
            line.getSecondPrototype().getPrototypeModel().eval()
            lines.append(line)
        return lines

    def computeLabels(self, inputs, line):

        prototype_1 = line.getFirstPrototype()
        prototype_2 = line.getSecondPrototype()

        output_1 = line.getFirstPrototype().getPrototypeModel()(inputs)
        output_2 = line.getSecondPrototype().getPrototypeModel()(inputs)

        # get distances from the prototypes for all inputs
        distances_1 = []
        distances_2 = []
        for i in range(inputs.shape[0]):
            distances_1.append(np.linalg.norm(inputs[i].detach().numpy() - prototype_1.getLocation().detach().numpy()))
            distances_2.append(np.linalg.norm(inputs[i].detach().numpy() - prototype_2.getLocation().detach().numpy()))

        distances_1 = torch.unsqueeze(torch.Tensor(np.array(distances_1)), 1)
        distances_2 = torch.unsqueeze(torch.Tensor(np.array(distances_2)), 1)

        # compute the weighted probability distribution
        outputs = output_1 / distances_1 + output_2 / distances_2

        # return the final weighted probability distribution
        return outputs