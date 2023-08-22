from training_datasets.GLUEMetaDataset import GLUEMetaDataset


class FewShotEpisodeBatchSampler:
    def __init__(self, dataset: GLUEMetaDataset, kShot, batchSize):
        super().__init__()
        self.dataset = dataset
        self.kShot = kShot * 2
        self.batchSize = batchSize
        self.nWays = []

    def __iter__(self):
        episodeBatch = []
        self.dataset.changeRandomIndex()
        for episode_i in range(self.batchSize):
            episode = next(iter(self.dataset))
            _, labels = episode
            self.nWays.append(len(set(labels)))
            episodeBatch.extend(episode)
            if (episode_i + 1) % self.batchSize == 0:
                yield episodeBatch

    def getCollateFunction(self):
        def collate(dataset):
            idx = 0
            batchedData = []
            batchedLabels = []
            while idx < len(dataset):
                data, labels = dataset[idx]
                labels = labels.tolist()
                # construct the support and query set
                supportSet = []
                supportLabels = []
                querySet = []
                queryLabels = []
                for i in range(len(labels)):
                    if supportLabels.count(labels[i]) < self.kShot // 2:
                        supportSet.append(data[i])
                        supportLabels.append(labels[i])
                    else:
                        querySet.append(data[i])
                        queryLabels.append(labels[i])
                data = supportSet + querySet
                labels = supportLabels + queryLabels
                batchedData.append(data)
                batchedLabels.append(labels)
                idx += 1
            return batchedData, batchedLabels, self.dataset.getCurrentTask()
        return collate
