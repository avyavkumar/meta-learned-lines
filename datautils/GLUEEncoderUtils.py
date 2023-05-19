import numpy as np
import torch

from transformers import BertTokenizer, BertModel
from random import randint

BERT_INPUT_DIMS = 768


def get_model():
    return BertModel.from_pretrained("bert-base-cased")


def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-cased")


"""
Return the episodic datautils with encoded data points.
If there are two sentences (for example in an entailment task), concatenate the sentences in the form
[CLS] <sentence_1> SEP <sentence_2> SEP
"""


def get_labelled_GLUE_episodic_training_data(episode):
    model = get_model()
    tokenizer = get_tokenizer()
    training_encodings = []
    training_labels = episode[1]
    print(training_labels)
    for i in range(len(training_labels)):
        inputs = tokenizer(episode[0][i], return_tensors="pt")
        outputs = model(**inputs)
        encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
        training_encodings.append(encoding)
    return torch.stack(training_encodings, dim=0), torch.Tensor(training_labels)
