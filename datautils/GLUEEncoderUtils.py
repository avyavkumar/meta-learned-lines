import numpy as np
import torch

from datasets import Dataset
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
def get_labelled_GLUE_episodic_training_data(dataset: Dataset):
    model = get_model()
    tokenizer = get_tokenizer()
    training_encodings = []
    training_labels = dataset['train']['label']
    if 'sentence1' in dataset['train'][0].keys():
        sentence1 = dataset['train']['sentence1']
        sentence2 = dataset['train']['sentence2']
        for i in range(len(sentence1)):
            concatenated_sentence = 'CLS' + sentence1[i] + 'SEP' + sentence2[i] + 'SEP'
            inputs = tokenizer(concatenated_sentence, return_tensors="pt")
            outputs = model(**inputs)
            encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
            training_encodings.append(encoding)
        return torch.stack(training_encodings, dim=0), torch.Tensor(training_labels)

    else:
        sentences = dataset['train']['sentence']
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs)
            encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
            training_encodings.append(encoding)
        return torch.stack(training_encodings, dim=0), torch.Tensor(training_labels)
