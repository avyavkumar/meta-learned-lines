import math
import random

from datasets import load_dataset
from torch.utils.data import Dataset

from training_datasets.GLUEDataset import GLUEDataset


class MultiTaskGLUEDataset(Dataset):
    def __init__(self, batchSize):
        self.taskNames = ['cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
        self.tasks = [GLUEDataset([load_dataset('glue', taskName)], 'train', -1) for taskName in self.taskNames]
        # for mnli and snli, split the datasets into two columns and make them learn against each other
        mnli = load_dataset('glue', 'mnli')
        for i in range(3):
            filteredMNLI = mnli.filter(lambda example: example['label'] != i)
            self.tasks.append(GLUEDataset([filteredMNLI], 'train', -1))
            self.taskNames.append("mnli" + str(i))
        snli = load_dataset('snli').filter(lambda example: example['label'] != -1)
        for i in range(3):
            filteredSNLI = snli.filter(lambda example: example['label'] != i)
            self.tasks.append(GLUEDataset([filteredSNLI], 'train', -1))
            self.taskNames.append("snli" + str(i))
        total = sum([math.sqrt(len(self.tasks[i])) for i in range(len(self.tasks))])
        self.probabilities = [math.sqrt(len(self.tasks[i])) / total for i in range(len(self.tasks))]
        self.distribution = []
        for idx, prob in enumerate(self.probabilities):
            for i in range(round(1000 * prob)):
                self.distribution.append(idx)
        random.shuffle(self.distribution)
        self.randomIndex = random.randint(0, 1000)
        self.batchSize = batchSize
        self.lengthItems = 0

    def __getitem__(self, i):
        self.lengthItems += 1
        if self.lengthItems == self.batchSize:
            self.randomIndex = random.randint(0, 1000)
        yield self.tasks[self.distribution[self.randomIndex]][random.randint(0, len(self.tasks[self.distribution[self.randomIndex]]))]

    def __len__(self):
        # return some random value as we don't depend on this method for sampling
        return 1000
