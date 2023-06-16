import random

from training_datasets.GLUEDataset import GLUEDataset


class FewShotEpisodeSampler:
    def __init__(self, dataset: GLUEDataset, kShot, nWay, shuffle=True):
        self.dataset = dataset
        # use twice the number of k shots for support and query set
        self.kShot = kShot * 2
        # IMPORTANT nWay should only be equal to all the number of classes, not just a subset#
        # torch accepts a list of indices from the main dataset#
        # therefore, we cannot remap non-zero indexed data points as this would require a change across the dataset
        # TODO check if the trainer has the code to remap labels
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
        # select subset of classes required for training
        requiredClassIndices = random.sample(classLabelIndices.keys(), self.nWay)
        for iteration in range(totalIterations):
            # sample 1 random batch each from all classes and construct support and query sets from each batch
            supportSet = []
            querySet = []
            for class_i in requiredClassIndices:
                labelIndices = random.sample(classLabelIndices[class_i], self.kShot)
                supportSet.extend(labelIndices[:self.kShot//2])
                querySet.extend(labelIndices[self.kShot//2:])
            # shuffle the support and query set
            random.shuffle(supportSet)
            random.shuffle(querySet)
            episode = supportSet + querySet
            yield episode

    def __len__(self):
        return sum(self.batchesPerClass.values()) // self.nWay
