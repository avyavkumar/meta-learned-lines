from samplers.FewShotValidationEpisodeSampler import FewShotValidationEpisodeSampler
from validation_datasets.ValidationDataset import ValidationDataset


class FewShotValidationEpisodeBatchSampler:
    def __init__(self, dataset: ValidationDataset, kShot):
        super().__init__()
        self.kShot = kShot * 2
        self.dataset = dataset
        self.sampler = FewShotValidationEpisodeSampler(dataset, kShot)
        self.batchSize = self.sampler.getBatchSize()

    def __iter__(self):
        episodeBatch = []
        for episode_i in range(self.batchSize):
            episodeBatch.extend(next(iter(self.sampler)))
            if (episode_i + 1) % self.batchSize == 0:
                yield episodeBatch

    def getCollateFunction(self):
        def collate(dataset):
            batchedData = []
            batchedLabels = []
            labelsList = self.dataset.getLabelsList()
            totalLabels = 0
            for i in range(len(labelsList)):
                totalLabels += len(labelsList[i])
            labelListIndices = {}
            i = 0
            for j in range(len(labelsList)):
                for k in range(len(labelsList[j])):
                    labelListIndices[i + k] = j
                i += len(labelsList[j])
            for i in range(len(labelsList)):
                batchedData.append([])
                batchedLabels.append([])
            for idx in range(len(dataset)):
                dataPoint, label = dataset[idx]
                batchedData[labelListIndices[label]].append(dataPoint)
                batchedLabels[labelListIndices[label]].append(label)
            return batchedData, batchedLabels
        return collate
