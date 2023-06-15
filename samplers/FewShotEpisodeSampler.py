import random

from training_datasets.GLUEDataset import GLUEDataset


class FewShotEpisodeSampler:
    def __init__(self, dataset: GLUEDataset, kShot, nWay, shuffle=True):
        self.dataset = dataset
        # use twice the number of k shots for support and query set
        self.kShot = kShot * 2
        self.nWay = nWay
        self.shuffle = shuffle

        # get random labels and sample a few-shot batch
        classLabelIndices = self.dataset.getClassLabelIndices()
        self.batchesPerClass = {}
        for class_i in classLabelIndices.keys():
            self.batchesPerClass[class_i] = len(classLabelIndices[class_i]) // self.kShot

    def __iter__(self):
        # it is important to change labels to ensure meta-learning generalisation in training
        if self.shuffle:
            self.dataset.reMapLabels()
        totalIterations = sum(self.batchesPerClass.values()) // self.nWay
        classLabelIndices = self.dataset.getClassLabelIndices()
        for iteration in range(totalIterations):
            # sample 1 random batch each from all classes and construct support and query sets from each batch
            supportSet = []
            querySet = []
            for class_i in classLabelIndices.keys():
                labelIndices = random.sample(classLabelIndices[class_i], self.kShot)
                supportSet.extend(labelIndices[:self.kShot])
                querySet.extend(labelIndices[self.kShot:])
            episode = supportSet + querySet
            yield episode

    def __len__(self):
        return sum(self.batchesPerClass.values()) // self.nWay
