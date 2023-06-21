class PrototypeModel:
    def getEncoder(self) -> str:
        pass

    def getEncodings(self, sentences):
        pass

    def scaleGradients(self, scalingFactor):
        pass

    def setParamsOfLinearLayer(self, weights, bias):
        pass

    def getPrototypicalEmbedding(self, inputs):
        pass
