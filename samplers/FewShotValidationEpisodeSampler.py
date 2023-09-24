import random

class FewShotValidationEpisodeSampler:
    def __init__(self, dataset, kShot):
        # use twice the number of k shots for support and query set
        self.kShot = kShot * 2
        self.dataset = dataset
        self.currentDataset = 0
        self.labelsList = dataset.getLabelsList()
        self.totalEpisodes = 1

    def __iter__(self):
        for iteration in range(self.totalEpisodes):
            # sample 1 random batch each from all classes and construct support and query sets from each batch
            supportSet = []
            querySet = []
            currentLabels = self.getCurrentLabels()
            classLabelIndices = self.constructClassLabelIndices(currentLabels)
            for class_i in classLabelIndices.keys():
                labelIndices = random.sample(classLabelIndices[class_i], self.kShot)
                supportSet.extend(labelIndices[:self.kShot//2])
                querySet.extend(labelIndices[self.kShot//2:])
            # shuffle the support and query set
            random.shuffle(supportSet)
            random.shuffle(querySet)
            episode = supportSet + querySet
            # select dataset required for training
            self.currentDataset += 1
            self.currentDataset = self.currentDataset % len(self.labelsList)
            yield episode

    def getCurrentLabels(self):
        totalLabels = 0
        currentLabels = []
        for i in range(self.currentDataset):
            totalLabels += len(self.labelsList[i])
        for i in range(len(self.labelsList[self.currentDataset])):
            currentLabels.append(totalLabels + i)
        return currentLabels

    def constructClassLabelIndices(self, currentLabels):
        classLabelIndices = {}
        _, transformedLabels = self.dataset.getData()
        for i, label in enumerate(transformedLabels, 0):
            if label in currentLabels:
                if label not in classLabelIndices:
                    classLabelIndices[label] = []
                classLabelIndices[label].append(i)
        return classLabelIndices

    def getBatchSize(self):
        return len(self.labelsList)
