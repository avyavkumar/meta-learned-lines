from datasets import load_dataset
import learn2learn as l2l
from training_datasets.GLUEDataset import GLUEDataset
from training_datasets.GLUEMetaDataset import GLUEMetaDataset


def train():
    cola = load_dataset('glue', 'cola')
    sst2 = load_dataset('glue', 'sst2')
    glue_dataset = GLUEDataset([cola, sst2], 'train')

    glue_meta_dataset = GLUEMetaDataset(glue_dataset, k=3, numTasks=5000)
    task = glue_meta_dataset.getTask()
