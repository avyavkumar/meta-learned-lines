from torch.utils.data import Dataset


class EncodingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.labels = labels
        self.encodings = encodings

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.encodings)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.encodings[index], self.labels[index]
