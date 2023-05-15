from prototypes.Prototype import Prototype
from prototypes.models.PrototypeClassifierModels import CLASSIFIER_MODEL_2NN, CLASSIFIER_MODEL_4NN, \
    PrototypeClassifierModel4NN, PrototypeClassifierModel2NN
from datautils.LEOPARDEncoderUtils import BERT_INPUT_DIMS


class Line:
    def __init__(self, centroids, labels, modelType: str, labelDict=None):
        self.centroids = centroids
        self.labels = labels
        self.firstPrototype = None
        self.secondPrototype = None
        self.modelType = modelType
        self.openPrototypes(modelType)
        self.labelDict = self.createLabelDict(labels, labelDict)

    def createLabelDict(self, labels, labelDict):
        if labelDict is not None:
            return labelDict
        lineLabelIndices = {}
        for i in range(len(labels)):
            lineLabelIndices[labels[i]] = i
        return lineLabelIndices

    def openPrototypes(self, modelType):
        classes = len(self.centroids)
        if modelType == CLASSIFIER_MODEL_4NN:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeClassifierModel4NN(BERT_INPUT_DIMS, classes))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeClassifierModel4NN(BERT_INPUT_DIMS, classes))
        elif modelType == CLASSIFIER_MODEL_2NN:
            self.firstPrototype = Prototype(self.centroids[0], PrototypeClassifierModel2NN(BERT_INPUT_DIMS, classes))
            self.secondPrototype = Prototype(self.centroids[-1], PrototypeClassifierModel2NN(BERT_INPUT_DIMS, classes))
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