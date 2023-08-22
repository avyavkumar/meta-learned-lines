import random
from collections import Counter

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
            if 'premise' in datasets[d_i][split][0].keys():
                premise = datasets[d_i][split]['premise']
                hypothesis = datasets[d_i][split]['hypothesis']
                labels = datasets[d_i][split]['label']
                for i in range(len(premise)):
                    concatenated_sentence = '[CLS] ' + premise[i] + ' [SEP] ' + hypothesis[i] + ' [SEP]'
                    self.sentences.append(concatenated_sentence)
                    self.labels.append(labels[i] + totalLabels)
            elif 'question1' in datasets[d_i][split][0].keys():
                question1 = datasets[d_i][split]['question1']
                question2 = datasets[d_i][split]['question2']
                labels = datasets[d_i][split]['label']
                for i in range(len(question1)):
                    concatenated_sentence = '[CLS] ' + question1[i] + ' [SEP] ' + question2[i] + ' [SEP]'
                    self.sentences.append(concatenated_sentence)
                    self.labels.append(labels[i] + totalLabels)
            elif 'sentence1' in datasets[d_i][split][0].keys():
                sentence1 = datasets[d_i][split]['sentence1']
                sentence2 = datasets[d_i][split]['sentence2']
                labels = datasets[d_i][split]['label']
                for i in range(len(sentence1)):
                    concatenated_sentence = '[CLS] ' + sentence1[i] + ' [SEP] ' + sentence2[i] + ' [SEP]'
                    self.sentences.append(concatenated_sentence)
                    self.labels.append(labels[i] + totalLabels)
            else:
                sentence = datasets[d_i][split]['sentence']
                labels = datasets[d_i][split]['label']
                for i in range(len(sentence)):
                    self.sentences.append('[CLS] ' + sentence[i] + ' [SEP]')
                    self.labels.append(labels[i] + totalLabels)
            totalLabels += len(set(datasets[d_i][split]['label']))
        for label_i in range(len(self.labels)):
            if self.labels[label_i] not in self.classLabelIndices:
                self.classLabelIndices[self.labels[label_i]] = []
                self.classLabelIndices[self.labels[label_i]].append(label_i)
            else:
                self.classLabelIndices[self.labels[label_i]].append(label_i)
        print(Counter(self.labels))
        self.balanceDataset()
        if length != -1:
            self.truncateDataset(length)
        print(Counter(self.labels))

    def truncateDataset(self, length):
        totalClasses = len(set(self.labels))
        totalValues = min(length, len(self.labels) // totalClasses)
        truncatedData = []
        truncatedLabels = []
        labelCount = {}
        for i in set(self.labels):
            labelCount[i] = 0
        for i in range(len(self.labels)):
            if labelCount[self.labels[i]] < totalValues:
                truncatedData.append(self.sentences[i])
                truncatedLabels.append((self.labels[i]))
                labelCount[self.labels[i]] += 1
        self.sentences = truncatedData
        self.labels = truncatedLabels

    def balanceDataset(self):
        counterLabels = Counter(self.labels)
        _, values = min(counterLabels.items(), key=lambda x: x[1])
        _, maxValues = max(counterLabels.items(), key=lambda x: x[1])
        if 0.9 < values / maxValues:
            return
        balancedData = []
        balancedLabels = []
        labelCount = {}
        for i in set(self.labels):
            labelCount[i] = 0
        for i in range(len(self.labels)):
            if labelCount[self.labels[i]] < values:
                balancedData.append(self.sentences[i])
                balancedLabels.append((self.labels[i]))
                labelCount[self.labels[i]] += 1
        self.sentences = balancedData
        self.labels = balancedLabels

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
