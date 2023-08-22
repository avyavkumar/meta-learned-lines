import math
import random
from collections import Counter

import learn2learn as l2l
from datasets import load_dataset
from torch.utils.data import Dataset

from training_datasets.GLUEDataset import GLUEDataset


class GLUEMetaDataset(Dataset):
    def __init__(self, k, numTasks, length=-1):
        self.taskNames = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
        self.tasks = [GLUEDataset([load_dataset('glue', taskName)], 'train', length) for taskName in self.taskNames]
        self.tasks.append(GLUEDataset([load_dataset('snli').filter(lambda example: example['label'] != -1)], 'train', length))
        self.taskNames.append('snli')
        self.nWays = [len(set(self.tasks[i].getLabels())) for i in range(len(self.taskNames))]
        total = sum([math.sqrt(len(self.tasks[i].getLabels())) for i in range(len(self.taskNames))])
        self.probabilities = [math.sqrt(len(self.tasks[i].getLabels()))/total for i in range(len(self.taskNames))]
        self.distribution = []
        for idx, prob in enumerate(self.probabilities):
            for i in range(round(1000*prob)):
                self.distribution.append(idx)
        random.shuffle(self.distribution)
        self.glueDatasets = [l2l.data.MetaDataset(task) for task in self.tasks]
        self.taskSet = []
        self.randomIndex = 0
        for i in range(len(self.taskNames)):
            transforms = [
                l2l.data.transforms.NWays(self.glueDatasets[i], n=len(set(self.glueDatasets[i][:][1]))),
                l2l.data.transforms.KShots(self.glueDatasets[i], k=k*2),
                l2l.data.transforms.LoadData(self.glueDatasets[i]),
                l2l.data.transforms.RemapLabels(self.glueDatasets[i]),
                l2l.data.transforms.ConsecutiveLabels(self.glueDatasets[i]),
            ]
            self.taskSet.append(l2l.data.TaskDataset(dataset=self.glueDatasets[i], task_transforms=transforms,
                                            num_tasks=numTasks))

    def changeRandomIndex(self):
        self.randomIndex = random.randint(0, len(self.distribution)-1)

    def getCurrentTask(self):
        return self.taskNames[self.distribution[self.randomIndex]]

    def getTask(self):
        return self.taskSet[self.distribution[self.randomIndex]].sample()

    def getTotalTasks(self):
        totalTasks = 0
        for i in range(len(self.taskNames)):
            totalTasks += self.taskSet[i].num_tasks
        return totalTasks

    def __iter__(self):
        yield self.getTask()

    def __len__(self):
        # we return a default value here as we need to run the validation episode after every few epochs
        return 100

    def __getitem__(self, index):
        return self.getTask()
