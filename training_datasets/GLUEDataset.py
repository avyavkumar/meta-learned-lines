import random

import datasets as huggingface_datasets
import torch
from torch.utils.data import Dataset


class GLUEDataset(Dataset):
    def __init__(self, datasets: [huggingface_datasets.Dataset], split, length=-1):
        self.sentences = []
        self.labels = []
        self.classLabelIndices = {}
        totalLabels = 0
        for d_i in range(len(datasets)):
            values = datasets[d_i][split].num_rows if length == -1 else length
            if 'premise' in datasets[d_i][split][0].keys():
                premise = datasets[d_i][split]['premise']
                hypothesis = datasets[d_i][split]['hypothesis']
                labels = datasets[d_i][split]['label']
                for i in range(len(premise)):
                    concatenated_sentence = '[CLS] ' + premise[i] + ' [SEP] ' + hypothesis[i] + ' [SEP]'
                    if i < values:
                        self.sentences.append(concatenated_sentence)
                        self.labels.append(labels[i] + totalLabels)
            elif 'question1' in datasets[d_i][split][0].keys():
                question1 = datasets[d_i][split]['question1']
                question2 = datasets[d_i][split]['question2']
                labels = datasets[d_i][split]['label']
                for i in range(len(question1)):
                    concatenated_sentence = '[CLS] ' + question1[i] + ' [SEP] ' + question2[i] + ' [SEP]'
                    if i < values:
                        self.sentences.append(concatenated_sentence)
                        self.labels.append(labels[i] + totalLabels)
            elif 'sentence1' in datasets[d_i][split][0].keys():
                sentence1 = datasets[d_i][split]['sentence1']
                sentence2 = datasets[d_i][split]['sentence2']
                labels = datasets[d_i][split]['label']
                for i in range(len(sentence1)):
                    concatenated_sentence = '[CLS] ' + sentence1[i] + ' [SEP] ' + sentence2[i] + ' [SEP]'
                    if i < values:
                        self.sentences.append(concatenated_sentence)
                        self.labels.append(labels[i] + totalLabels)
            else:
                sentence = datasets[d_i][split]['sentence']
                labels = datasets[d_i][split]['label']
                for i in range(len(sentence)):
                    if i < values:
                        self.sentences.append('[CLS] ' + sentence[i] + ' [SEP]')
                        self.labels.append(labels[i] + totalLabels)
            totalLabels += len(set(datasets[d_i][split]['label']))
        for label_i in range(len(self.labels)):
            if self.labels[label_i] not in self.classLabelIndices:
                self.classLabelIndices[self.labels[label_i]] = []
                self.classLabelIndices[self.labels[label_i]].append(label_i)
            else:
                self.classLabelIndices[self.labels[label_i]].append(label_i)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.sentences)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.sentences[index], self.labels[index]

    def getClassLabelIndices(self):
        return self.classLabelIndices

    def getLabels(self):
        return self.labels

    def reMapLabels(self):
        # step 1 - get a random permutation of the labels and re-write self.classLabelIndices
        totalClasses = len(self.classLabelIndices.keys())
        listOfClasses = [i for i in range(totalClasses)]
        random.shuffle(listOfClasses)
        class_i = 0
        updatedDictOfLabelIndices = {}
        for classLabel in self.classLabelIndices.keys():
            updatedDictOfLabelIndices[listOfClasses[class_i]] = self.classLabelIndices[classLabel]
            class_i += 1
        self.classLabelIndices = updatedDictOfLabelIndices
        # step 2 - update self.labels to reflect new values
        for classLabel in self.classLabelIndices.keys():
            for i in range(len(self.classLabelIndices[classLabel])):
                self.labels[self.classLabelIndices[classLabel][i]] = classLabel
