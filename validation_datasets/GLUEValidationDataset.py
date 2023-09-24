from datasets import load_dataset
from torch.utils.data import Dataset

from datautils.ValidationDataUtils import get_validation_data
from training_datasets.GLUEDataset import GLUEDataset


class GLUEValidationDataset(Dataset):
    def __init__(self):
        print("USing GLUEValidationDataset...")
        self.taskNames = ['cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
        self.tasks = [GLUEDataset([load_dataset('glue', taskName)], 'validation') for taskName in self.taskNames]
        self.tasks.append(GLUEDataset([load_dataset('snli').filter(lambda example: example['label'] != -1)], 'validation'))
        self.taskNames.append('snli')
        self.tasks.append(GLUEDataset([load_dataset('glue', 'mnli')], ['validation_matched', 'validation_mismatched']))
        self.taskNames.append('mnli')
        self.sentences, self.labels = self.getSentencesAndLabels()
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

    def getSentencesAndLabels(self):
        sentences = []
        labels = []
        for task in self.tasks:
            taskSentences = []
            taskLabels = []
            for i in range(len(task)):
                sentence, label = task[i]
                taskSentences.append(sentence)
                taskLabels.append(label)
            sentences.append(taskSentences)
            labels.append(taskLabels)
        return sentences, labels

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
