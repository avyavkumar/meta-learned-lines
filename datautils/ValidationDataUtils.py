import os
import json
import random

RATINGS_DATA_PATH = "validation_data/rating_categorisation"
SENTIMENT_DATA_PATH = "validation_data/sentiment_classification"


def get_categories_ratings():
    categories = os.listdir(RATINGS_DATA_PATH)
    return categories


def get_categories_sentiment():
    categories = os.listdir(SENTIMENT_DATA_PATH)
    return categories


def get_labelled_validation_sentences(path, category, balance_and_subsample=True):
    sentences = []
    labels = []
    label_keys = {}
    label_index = 0
    data_path = path + "/" + category + "/"
    for file_name in os.listdir(data_path):
        if file_name.endswith("data.json"):
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
    if balance_and_subsample is True:
        return balance_and_subsample_dataset(sentences, labels)
    else:
        return sentences, labels

def balance_and_subsample_dataset(sentences, labels):
    total_classes = len(set(labels))
    elements_per_class = 20 // total_classes
    subsampled_sentences = []
    subsampled_labels = []
    for class_i in range(total_classes):
        elements = []
        for i in range(len(labels)):
            if labels[i] == class_i:
                elements.append(i)
        indices_required = random.sample(elements, elements_per_class)
        for index in indices_required:
            subsampled_sentences.append(sentences[index])
            subsampled_labels.append(labels[index])
    return subsampled_sentences, subsampled_labels


def get_validation_data():
    sentences = []
    labels = []
    # for category in get_categories_ratings():
    #     cat_sentences, cat_labels = get_labelled_validation_sentences(RATINGS_DATA_PATH, category)
    #     sentences.append(cat_sentences)
    #     labels.append(cat_labels)
    for category in get_categories_sentiment():
        cat_sentences, cat_labels = get_labelled_validation_sentences(SENTIMENT_DATA_PATH, category)
        sentences.append(cat_sentences)
        labels.append(cat_labels)
    return sentences, labels
