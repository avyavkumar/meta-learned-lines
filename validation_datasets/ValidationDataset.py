from torch.utils.data import Dataset

from datautils.ValidationDataUtils import get_validation_data


class ValidationDataset(Dataset):
    def __init__(self):
        self.sentences, self.labels = get_validation_data()
        # store a dictionary of keys for lookup and transform the data into continuous labels
        self.labelsList = [list(set(labels)) for labels in self.labels]
        totalLabels = 0
        self.transformedSentences = []
        self.transformedLabels = []
        for i in range(len(self.sentences)):
            for sentence_i in range(len(self.sentences[i])):
                self.transformedSentences.append(self.sentences[i][sentence_i])
                self.transformedLabels.append(self.labels[i][sentence_i] + totalLabels)
            totalLabels += len(set(self.labels[i]))

    def getLabelsList(self):
        return self.labelsList

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.transformedSentences)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.transformedSentences[index], self.transformedLabels[index]

    def getData(self):
        return self.transformedSentences, self.transformedLabels
