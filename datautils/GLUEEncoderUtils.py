import torch
from utils.ModelUtils import TOKENIZER, MODEL, DEVICE

BERT_INPUT_DIMS = 768

"""
Return the episodic datautils with encoded data points.
If there are two sentences (for example in an entailment task), concatenate the sentences in the form
[CLS] <sentence_1> SEP <sentence_2> SEP
"""


def get_labelled_episodic_training_data(data, labels):
    training_encodings = []
    training_labels = labels
    for i in range(len(training_labels)):
        inputs = TOKENIZER(data[i], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        outputs = MODEL(**inputs)
        encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
        training_encodings.append(encoding.detach().cpu())
        del inputs, outputs, encoding
    return torch.stack(training_encodings, dim=0), torch.Tensor(training_labels)
