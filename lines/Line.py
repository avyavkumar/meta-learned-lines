from torch import nn

from prototypes.Prototype import Prototype
from prototypes.models.PrototypeClassifierModels import CLASSIFIER_MODEL_2NN, CLASSIFIER_MODEL_4NN, \
    PrototypeClassifierModel4NN, PrototypeClassifierModel2NN
from prototypes.models.PrototypeMetaLinearModel import PrototypeMetaLinearModel
from utils.Constants import BERT_DIMS, PROTOTYPE_META_MODEL, HIDDEN_MODEL_SIZE


class Line:
    def __init__(self, centroids, labels, modelType: str, metaLearner=None, labelDict=None):
        self.centroids = centroids
        self.labels = labels
        self.firstPrototype = None
        self.secondPrototype = None
        self.modelType = modelType
        self.openPrototypes(modelType, metaLearner)
        self.labelDict = self.createLabelDict(labels, labelDict)

    def createLabelDict(self, labels, labelDict):
        if labelDict is not None:
            return labelDict
        lineLabelIndices = {}
        for i in range(len(labels)):
            lineLabelIndices[labels[i]] = i
        return lineLabelIndices

    def openPrototypes(self, modelType, metaLearner=None):
        classes = len(self.centroids)
        if modelType == CLASSIFIER_MODEL_4NN:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeClassifierModel4NN(BERT_DIMS, classes))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeClassifierModel4NN(BERT_DIMS, classes))
        elif modelType == CLASSIFIER_MODEL_2NN:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeClassifierModel2NN(BERT_DIMS, classes))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeClassifierModel2NN(BERT_DIMS, classes))
        elif modelType == PROTOTYPE_META_MODEL:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeMetaLinearModel(metaLearner, classes))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeMetaLinearModel(metaLearner, classes))
        else:
            raise ValueError("Unknown encoder type", modelType, "for creating a soft-label prototype")

    def getCentroids(self):
        return self.centroids

    def getLabels(self):
        return self.labels

    def getModelType(self):
        return self.modelType

    def getFirstPrototype(self) -> Prototype:
        return self.firstPrototype

    def getSecondPrototype(self) -> Prototype:
        return self.secondPrototype

    def getLabelDict(self):
        return self.labelDict
