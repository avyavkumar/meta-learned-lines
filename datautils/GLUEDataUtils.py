from random import randint

from datautils.GLUEUtils import get_glue_tasks, list_glue_tasks

EPSILON = 1e-5

def get_training_split(fraction = 1.0):
    glue_datasets = get_glue_tasks()
    # split the datautils into training and test partitions without shuffling for reproducibility
    # discard the test partition and return the train partition
    glue_datasets = [glue_datasets[i]['train'].train_test_split(test_size=1-fraction+EPSILON)['train'] for i in range(len(glue_datasets))]
    return glue_datasets

# def get_validation_splt():
#     glue_datasets = get_glue_tasks()
#     glue_datasets = [glue_datasets[i]['validation'] for i in range(len(glue_datasets))]
#     return glue_datasets
#
# def get_test_split():
#     glue_datasets = get_glue_tasks()
#     glue_datasets = [glue_datasets[i]['test'] for i in range(len(glue_datasets))]
#     return glue_datasets

def get_random_GLUE_dataset():
    return get_glue_tasks()[randint(0, len(list_glue_tasks()))]

def combine_datasets(datasets: list):
    pass
