import torch

from lines.Line import Line
import numpy as np

from datautils.LEOPARDEncoderUtils import get_labelled_centroids
from lines.lo_shot_utils import find_lines_R_multiD


class LineGenerator:
    def __init__(self, trainingSet, modelType: str):
        self.trainingSet = trainingSet
        self.modelType = modelType

    def generateLines(self, metaLearner=None):
        training_encodings = self.trainingSet['encodings']
        training_labels = self.trainingSet['labels']
        total_classes = len(set(training_labels.tolist()))

        centroids, centroid_labels = get_labelled_centroids(training_encodings, training_labels.tolist())
        k = len(centroid_labels)
        # invoke Ilia's code
        dims = training_encodings[0].shape[0]
        lines_generated = find_lines_R_multiD(training_encodings.detach().cpu().numpy(), training_labels.tolist(), centroids.detach().cpu().numpy(), dims, k - 1)
        lines = []
        for i in range(len(lines_generated)):
            centroids_required = []
            centroid_labels_required = []
            for j in lines_generated[i]:
                centroids_required.append(centroids[j])
                centroid_labels_required.append(centroid_labels[j])
            lines.append(Line(total_classes, torch.stack(centroids_required, dim=0), np.array(centroid_labels_required), self.modelType, metaLearner=metaLearner))
        return lines
