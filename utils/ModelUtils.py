import torch
from transformers import BertModel, BertTokenizer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CPU_DEVICE = torch.device('cpu')
MODEL = BertModel.from_pretrained("bert-base-cased").to(DEVICE)
TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")

def get_prototypes(inputs, labels):
    unique_labels, _ = torch.unique(labels).sort()
    prototypes = []
    for label in unique_labels:
        prototypes.append(inputs[torch.where(labels == label)[0]].mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)
    return prototypes.to(DEVICE), unique_labels.to(DEVICE)
