import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path

from datautils.LEOPARDDataUtils import get_labelled_training_sentences, get_labelled_test_sentences, \
    get_labelled_validation_sentences, get_categories, get_labelled_test_sentences_entity_typing, \
    get_labelled_training_sentences_entity_typing
from utils.ModelUtils import DEVICE


def get_model():
    return BertModel.from_pretrained("bert-base-cased").to(DEVICE)

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-cased")

def get_labelled_LEOPARD_training_data(category, shot, episode):
    sentences, training_labels, label_keys = get_labelled_training_sentences(category, shot, episode)
    training_encodings = []
    model = get_model()
    tokenizer = get_tokenizer()
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
        training_encodings.append(encoding)
    return sentences, torch.stack(training_encodings, dim=0), np.array(training_labels), label_keys

def get_labelled_LEOPARD_training_data_meta_encoded(metaLearner, category, shot, episode):
    with torch.no_grad():
        sentences, training_labels, label_keys = get_labelled_training_sentences(category, shot, episode)
        training_encodings = []
        for sentence in sentences:
            encoding = metaLearner(sentence).reshape(-1)
            training_encodings.append(encoding)
        return sentences, torch.stack(training_encodings, dim=0), np.array(training_labels), label_keys

def get_labelled_centroids(training_encodings, training_labels):
    centroids = []
    centroid_labels = []
    for label in set(training_labels):
        centroids_per_label = []
        for i in range(len(training_labels)):
            if training_labels[i] == label:
                centroids_per_label.append(training_encodings[i])
        centroid = torch.mean(torch.stack(centroids_per_label, dim=0), dim=0)
        centroids.append(centroid)
        centroid_labels.append(label)
    return torch.stack(centroids, dim=0), np.array(centroid_labels)

def get_length_LEOPARD_categorical_test_data(category):
    _, test_labels = get_labelled_test_sentences(category)
    return len(test_labels)

def get_labelled_LEOPARD_test_data(category):
    sentences, test_labels = get_labelled_test_sentences(category)
    test_encodings = []
    model = get_model()
    tokenizer = get_tokenizer()
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
        test_encodings.append(encoding)
    return test_encodings, test_labels

def get_labelled_validation_data(category):
    sentences, test_labels = get_labelled_validation_sentences(category)
    test_encodings = []
    model = get_model()
    tokenizer = get_tokenizer()
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        encoding = outputs.last_hidden_state[:, 0, :].reshape(-1)
        test_encodings.append(encoding)
    return test_encodings, test_labels

def write_test_data(model):
    for category in get_categories():
        sentences, test_labels = get_labelled_test_sentences(category)
        data_path = "test_data/bert/" + category + "/"
        Path(data_path).mkdir(parents=True, exist_ok=True)
        for i in range(len(sentences)):
            label = test_labels[i]
            encoding_path = data_path + str(label)
            Path(encoding_path).mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                encoding = model(sentences[i]).reshape(-1)
                with open(encoding_path + "/" + str(i), 'wb+') as f:
                    np.save(f, encoding.detach().cpu().numpy())

def read_test_data(category):
    data_path = "test_data/bert/" + category + "/"
    encodings = []
    labels = []
    for label in os.listdir(data_path):
        for encoding in os.listdir(data_path + label):
            encoding_path = data_path + label + "/" + encoding
            with open(encoding_path, 'rb+') as f:
                encodings.append(torch.Tensor(np.load(f)))
                labels.append(label)
    return encodings, labels

def get_labelled_LEOPARD_training_data_entity_typing(category, shot, episode):
    sentences, entities, training_labels, label_keys = get_labelled_training_sentences_entity_typing(category, shot, episode)
    tokenizer = get_tokenizer()
    model = get_model()

    training_encodings = []
    for sentence, entity, label in zip(sentences, entities, training_labels):
        inputs = tokenizer(entity, return_tensors="pt")
        required_bert_tokens = [e for e in inputs['input_ids'].tolist()[0] if e not in [101, 102]]
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)

        inputs_indices_sentence = []
        for i in range(len(inputs['input_ids'].tolist()[0])):
            if inputs['input_ids'].tolist()[0][i] in required_bert_tokens:
                required_bert_tokens.remove(inputs['input_ids'].tolist()[0][i])
                inputs_indices_sentence.append(i)
        encodings = []
        for inputs_index_sentence in inputs_indices_sentence:
            encodings.append(outputs.last_hidden_state[:, inputs_index_sentence, :].detach().numpy())
        encoding = np.stack(encodings).mean(axis=0).reshape(-1)
        training_encodings.append(encoding)

    return torch.Tensor(np.array(training_encodings)), np.array(training_labels), label_keys

def get_labelled_LEOPARD_test_data_entity_typing(category):
    sentences, entities, test_labels = get_labelled_test_sentences_entity_typing(category)
    tokenizer = get_tokenizer()
    model = get_model()

    test_encodings = []
    for sentence, entity, label in zip(sentences, entities, test_labels):
        inputs = tokenizer(entity, return_tensors="pt")
        required_bert_tokens = [e for e in inputs['input_ids'].tolist()[0] if e not in [101, 102]]
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)

        inputs_indices_sentence = []
        for i in range(len(inputs['input_ids'].tolist()[0])):
            if inputs['input_ids'].tolist()[0][i] in required_bert_tokens:
                required_bert_tokens.remove(inputs['input_ids'].tolist()[0][i])
                inputs_indices_sentence.append(i)
        encodings = []
        for inputs_index_sentence in inputs_indices_sentence:
            encodings.append(outputs.last_hidden_state[:, inputs_index_sentence, :].detach().numpy())
        encoding = np.stack(encodings).mean(axis=0).reshape(-1)
        test_encodings.append(encoding)

    return test_encodings, test_labels

def write_test_data_entity_typing(category):
    sentences, entities, test_labels = get_labelled_test_sentences_entity_typing(category)
    tokenizer = get_tokenizer()
    model = get_model()

    data_path = "test_data/bert/" + category + "/"
    x = 0
    for sentence, entity, label in zip(sentences, entities, test_labels):
        inputs = tokenizer(entity, return_tensors="pt")
        required_bert_tokens = [e for e in inputs['input_ids'].tolist()[0] if e not in [101, 102]]
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)

        inputs_indices_sentence = []
        for i in range(len(inputs['input_ids'].tolist()[0])):
            if inputs['input_ids'].tolist()[0][i] in required_bert_tokens:
                required_bert_tokens.remove(inputs['input_ids'].tolist()[0][i])
                inputs_indices_sentence.append(i)
        encodings = []
        for inputs_index_sentence in inputs_indices_sentence:
            encodings.append(outputs.last_hidden_state[:, inputs_index_sentence, :].detach().numpy())
        encoding = np.stack(encodings).mean(axis=0).reshape(-1)

        # save the encoding
        encoding_path = data_path + str(label)
        Path(encoding_path).mkdir(parents=True, exist_ok=True)
        with open(encoding_path + "/" + str(x), 'wb+') as f:
            np.save(f, encoding)
        x += 1
