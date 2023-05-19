from lines.Line import Line
import numpy as np

from datautils.LEOPARDEncoderUtils import get_labelled_centroids
from utils.Constants import BERT_DIMS
from lines.lo_shot_utils import find_lines_R_multiD


class LineGenerator:
    def __init__(self, trainingSet, modelType: str):
        self.trainingSet = trainingSet
        self.modelType = modelType

    def generateLines(self, metaLearner=None):
        training_encodings = self.trainingSet['encodings']
        training_labels = self.trainingSet['labels']

        centroids, centroid_labels = get_labelled_centroids(training_encodings, training_labels.tolist())
        k = len(centroid_labels)
        print("computing lines now")
        # invoke Ilia's code
        lines_generated = find_lines_R_multiD(training_encodings.detach().numpy(), training_labels.tolist(),
                                              centroids.detach().numpy(), BERT_DIMS, k - 1)
        print("Lines have been computed")
        lines = []
        for i in range(len(lines_generated)):
            centroids_required = []
            centroid_labels_required = []
            for j in lines_generated[i]:
                centroids_required.append(centroids[j])
                centroid_labels_required.append(centroid_labels[j])
            lines.append(Line(centroids_required, np.array(centroid_labels_required), self.modelType, metaLearner=metaLearner))

        return lines
