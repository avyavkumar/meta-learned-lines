import os
import json

DATA_PATH = "leopard/data/json"

def get_categories():
    categories = os.listdir(DATA_PATH)
    categories.remove("restaurant")
    categories.remove("conll")
    categories.remove("sentiment_electronics")
    return categories

def get_labelled_training_sentences(category, shot, episode):
    sentences = []
    labels = []
    label_keys = {}
    label_index = 0
    data_path = DATA_PATH + "/" + category + "/"
    for file_name in os.listdir(data_path):
        if file_name.endswith("_" + str(episode) + "_" + str(shot) + ".json"):
            data = json.load(open(data_path + file_name))
            for index in range(len(data)):
                processed_sentence = data[index]['processed_sent']
                processed_sentence = processed_sentence.replace('[CLS]', '')
                processed_sentence = processed_sentence.replace('[SEP]', '')
                label = data[index]['label']
                sentences.append(processed_sentence)
                # convert categorical labels to numeric values
                if label not in label_keys:
                    label_keys[label] = label_index
                    label_index += 1
                labels.append(label_keys[label])
    return sentences, labels, label_keys

def get_labelled_test_sentences(category):
    sentences = []
    labels = []
    data_path = DATA_PATH + "/" + category + "/"
    for file_name in os.listdir(data_path):
        if file_name.endswith("_eval.json"):
            data = json.load(open(data_path + file_name))
            for index in range(len(data)):
                processed_sentence = data[index]['processed_sent']
                processed_sentence = processed_sentence.replace('[CLS]', '')
                processed_sentence = processed_sentence.replace('[SEP]', '')
                label = data[index]['label']
                sentences.append(processed_sentence)
                labels.append(label)
    return sentences[:5], labels[:5]

def get_labelled_validation_sentences(category):
    sentences = []
    labels = []
    data_path = DATA_PATH + "/" + category + "/"
    for file_name in os.listdir(data_path):
        if file_name.endswith("_eval.json"):
            data = json.load(open(data_path + file_name))
            for index in range(len(data)):
                processed_sentence = data[index]['processed_sent']
                processed_sentence = processed_sentence.replace('[CLS]', '')
                processed_sentence = processed_sentence.replace('[SEP]', '')
                label = data[index]['label']
                sentences.append(processed_sentence)
                labels.append(label)
    return sentences[:300], labels[:300]

def get_categories_entity_typing():
    categories = ["restaurant", "conll"]
    return categories

def get_labelled_training_sentences_entity_typing(category, shot, episode):
    sentences = []
    labels = []
    entities = []
    label_keys = {}
    label_index = 0
    data_path = DATA_PATH + "/" + category + "/"
    for file_name in os.listdir(data_path):
        if file_name.endswith("_" + str(episode) + "_" + str(shot) + ".json"):
            data = json.load(open(data_path + file_name))
            for index in range(len(data)):
                sentence_1 = data[index]['sentence1']
                entity = data[index]['sentence2']
                label = data[index]['label']
                sentences.append(sentence_1)
                entities.append(entity)
                # convert categorical labels to numeric values
                if label not in label_keys:
                    label_keys[label] = label_index
                    label_index += 1
                labels.append(label_keys[label])
    return sentences, entities, labels, label_keys

def get_labelled_test_sentences_entity_typing(category):
    sentences = []
    labels = []
    entities = []
    data_path = DATA_PATH + "/" + category + "/"
    for file_name in os.listdir(data_path):
        if file_name.endswith("_eval.json"):
            data = json.load(open(data_path + file_name))
            for index in range(len(data)):
                sentence_1 = data[index]['sentence1']
                entity = data[index]['sentence2']
                label = data[index]['label']
                sentences.append(sentence_1)
                labels.append(label)
                entities.append(entity)

    return sentences, entities, labels
