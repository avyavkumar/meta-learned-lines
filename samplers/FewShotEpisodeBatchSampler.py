from samplers.FewShotEpisodeSampler import FewShotEpisodeSampler
from training_datasets.GLUEDataset import GLUEDataset


class FewShotEpisodeBatchSampler:
    def __init__(self, dataset: GLUEDataset, kShot, nWay, batchSize, shuffle=True):
        super().__init__()
        self.nWay = nWay
        self.kShot = kShot * 2
        self.sampler = FewShotEpisodeSampler(dataset, kShot, nWay, shuffle)
        self.batchSize = batchSize

    def __iter__(self):
        episodeBatch = []
        for episode_i in range(self.batchSize):
            episodeBatch.extend(next(iter(self.sampler)))
            if (episode_i + 1) % self.batchSize == 0:
                yield episodeBatch

    def getCollateFunction(self):
        def collate(dataset):
            idx = 0
            batchedData = []
            batchedLabels = []
            while idx < len(dataset):
                data = []
                labels = []
                for i in range(self.nWay * self.kShot):
                    dataPoint, label = dataset[idx + i]
                    data.append(dataPoint)
                    labels.append(label)
                batchedData.append(data)
                batchedLabels.append(labels)
                idx += self.nWay * self.kShot
            return batchedData, batchedLabels
        return collate
