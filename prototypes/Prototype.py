from prototypes.PrototypeModel import PrototypeModel


class Prototype:
    def __init__(self, location, prototypeModel: PrototypeModel):
        self.location = location
        self.prototypeModel = prototypeModel

    def getLocation(self):
        return self.location

    def setLocation(self, location):
        self.location = location

    def getPrototypeModel(self):
        return self.prototypeModel

    def setPrototypeModel(self, prototypeModel):
        self.prototypeModel = prototypeModel
