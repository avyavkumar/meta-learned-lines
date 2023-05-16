import datasets as huggingface_datasets
from torch.utils.data import Dataset


class GLUEDataset(Dataset):
    def __init__(self, datasets: [huggingface_datasets.Dataset], split):
        self.sentences = []
        self.labels = []
        totalLabels = 0
        for d_i in range(len(datasets)):
            if 'sentence1' in datasets[d_i][split][0].keys():
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

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.sentences)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.sentences[index], self.labels[index]
