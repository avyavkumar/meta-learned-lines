from lines.Line import Line
import numpy as np

from datautils.LEOPARDEncoderUtils import get_labelled_centroids, BERT_INPUT_DIMS
from lines.lo_shot_utils import find_lines_R_multiD


class LineGenerator:
    def __init__(self, trainingSet, modelType: str):
        self.trainingSet = trainingSet
        self.modelType = modelType

    def generateLines(self):
        training_encodings = self.trainingSet['encodings']
        training_labels = self.trainingSet['labels']

        centroids, centroid_labels = get_labelled_centroids(training_encodings, training_labels)
        k = len(centroid_labels)

        # invoke Ilia's code
        lines_generated = find_lines_R_multiD(training_encodings.detach().numpy(), training_labels,
                                              centroids.detach().numpy(), BERT_INPUT_DIMS, k - 1)

        lines = []
        for i in range(len(lines_generated)):
            centroids_required = []
            centroid_labels_required = []
            for j in lines_generated[i]:
                centroids_required.append(centroids[j])
                centroid_labels_required.append(centroid_labels[j])
            lines.append(Line(centroids_required, np.array(centroid_labels_required), self.modelType))

        return lines
