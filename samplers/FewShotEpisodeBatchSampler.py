import torch

from samplers.FewShotEpisodeSampler import FewShotEpisodeSampler
from training_datasets.GLUEDataset import GLUEDataset


class FewShotEpisodeBatchSampler:
    def __init__(self, dataset: GLUEDataset, kShot, nWay, batchSize, shuffle=True):
        super().__init__()
        self.sampler = FewShotEpisodeSampler(dataset, kShot, nWay, shuffle)
        self.batchSize = batchSize

    def __iter__(self):
        episodeBatch = []
        for episode_i, episode in enumerate(self.sampler):
            episodeBatch.extend(episode)
            if (episode_i + 1) % self.batchSize == 0:
                yield episodeBatch
                episodeBatch = []

    def __len__(self):
        return len(self.sampler) // self.batchSize

    def getCollateFunction(self):
        def collate(dataset):
            idx = 0
            batchedData = []
            batchedLabels = []
            while idx < self.batchSize:
                data, labels = dataset[idx * self.batchSize: (idx + 1) * self.batchSize]
                batchedData.append(list(data))
                batchedLabels.append(list(labels))
                idx += 1
            return list(zip(batchedData, batchedLabels))
        return collate
