import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_prototypes(inputs, labels):
    labels = torch.tensor(labels, dtype=torch.int8)
    unique_labels, _ = torch.unique(labels).sort()
    prototypes = []
    for label in unique_labels:
        prototypes.append(inputs[torch.where(labels == label)[0]].mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)
    return prototypes, unique_labels
