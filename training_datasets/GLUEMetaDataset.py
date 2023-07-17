import math
import random

import learn2learn as l2l
from datasets import load_dataset
from torch.utils.data import Dataset

from training_datasets.GLUEDataset import GLUEDataset


class GLUEMetaDataset(Dataset):
    def __init__(self, k, numTasks):
        self.taskNames = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
        self.tasks = [GLUEDataset([load_dataset('glue', taskName)], 'train') for taskName in self.taskNames]
        self.nWays = [len(set(self.tasks[i].getLabels())) for i in range(len(self.taskNames))]
        total = sum([math.sqrt(len(self.tasks[i].getLabels())) for i in range(len(self.taskNames))])
        self.probabilities = [math.sqrt(len(self.tasks[i].getLabels()))/total for i in range(len(self.taskNames))]
        self.distribution = []
        for idx, prob in enumerate(self.probabilities):
            for i in range(round(100*prob)):
                self.distribution.append(idx)
        random.shuffle(self.distribution)
        self.glueDatasets = [l2l.data.MetaDataset(task) for task in self.tasks]
        self.taskSet = []
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

    def getTask(self):
        return self.taskSet[self.distribution[random.randint(0, 99)]].sample()

    def getTotalTasks(self):
        totalTasks = 0
        for i in range(len(self.taskNames)):
            totalTasks += self.taskSet[i].num_tasks
        return totalTasks

    def __iter__(self):
        yield self.getTask()

    def __len__(self):
        return self.getTotalTasks()

    def __getitem__(self, index):
        return self.getTask()
