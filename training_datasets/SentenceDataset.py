from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels):
        self.labels = labels
        self.sentences = sentences

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.sentences)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.sentences[index], self.labels[index]
