from torch import nn

from prototypes.Prototype import Prototype
from prototypes.models.PrototypeClassifierModels import CLASSIFIER_MODEL_2NN, CLASSIFIER_MODEL_4NN, \
    PrototypeClassifierModel4NN, PrototypeClassifierModel2NN
from prototypes.models.PrototypeMetaLinearModel import PrototypeMetaLinearModel
from utils import ModelUtils
from utils.Constants import BERT_DIMS, PROTOTYPE_META_MODEL, HIDDEN_MODEL_SIZE


class Line:
    def __init__(self, totalClasses, centroids, labels, modelType: str, metaLearner=None):
        self.totalClasses = totalClasses
        self.centroids = centroids
        self.labels = labels
        self.firstPrototype = None
        self.secondPrototype = None
        self.modelType = modelType
        self.openPrototypes(modelType, metaLearner)

    def openPrototypes(self, modelType, metaLearner=None):
        if modelType == CLASSIFIER_MODEL_4NN:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeClassifierModel4NN(BERT_DIMS, self.totalClasses))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeClassifierModel4NN(BERT_DIMS, self.totalClasses))
        elif modelType == CLASSIFIER_MODEL_2NN:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeClassifierModel2NN(BERT_DIMS, self.totalClasses))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeClassifierModel2NN(BERT_DIMS, self.totalClasses))
        elif modelType == PROTOTYPE_META_MODEL:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeMetaLinearModel(metaLearner, self.totalClasses))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeMetaLinearModel(metaLearner, self.totalClasses))
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

    # def getLabelDict(self):
    #     return self.labelDict
