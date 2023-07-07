from samplers.FewShotValidationEpisodeSampler import FewShotValidationEpisodeSampler
from validation_datasets.ValidationDataset import ValidationDataset


class FewShotValidationEpisodeBatchSampler:
    def __init__(self, dataset: ValidationDataset, kShot, totalBatches=10):
        super().__init__()
        self.kShot = kShot * 2
        self.dataset = dataset
        self.sampler = FewShotValidationEpisodeSampler(dataset, kShot)
        self.batchSize = self.sampler.getBatchSize()
        self.totalBatches = totalBatches

    def __iter__(self):
        episodeBatch = []
        for batch in range(self.totalBatches):
            for episode_i in range(self.batchSize):
                episodeBatch.extend(next(iter(self.sampler)))
            if (batch + 1) % self.totalBatches == 0:
                yield episodeBatch

    def getCollateFunction(self):
        def collate(dataset):
            labelsList = self.dataset.getLabelsList()
            labelListIndices = {}
            i = 0
            for j in range(len(labelsList)):
                for k in range(len(labelsList[j])):
                    labelListIndices[i + k] = j
                i += len(labelsList[j])
            dataset_i = 0
            data = []
            labels = []
            while dataset_i < len(dataset):
                batchedData = []
                batchedLabels = []
                for i in range(len(labelsList)):
                    batchedData.append([])
                    batchedLabels.append([])
                for idx in range(len(labelListIndices.keys()) * self.kShot):
                    dataPoint, label = dataset[dataset_i + idx]
                    batchedData[labelListIndices[label]].append(dataPoint)
                    batchedLabels[labelListIndices[label]].append(label)
                dataset_i += len(labelListIndices.keys()) * self.kShot
                data.extend(batchedData)
                labels.extend(batchedLabels)
            return data, labels
        return collate
