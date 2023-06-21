from torch.utils.data import Dataset


class SentenceEncodingDataset(Dataset):
    def __init__(self, sentences, encodings, labels):
        self.labels = labels
        self.encodings = encodings
        self.sentences = sentences

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.sentences)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.sentences[index], self.encodings[index], self.labels[index]
