import os

from datautils.LEOPARDDataUtils import get_categories
from evaluation.Evaluator import Evaluator
from evaluation.MetaClassifier import MetaClassifier
from prototypes.models.PrototypeClassifierModels import CLASSIFIER_MODEL_2NN
from datautils.LEOPARDEncoderUtils import get_labelled_LEOPARD_training_data
from lines.LineGenerator import LineGenerator
from utils.Constants import PROTOTYPE_META_MODEL

test_params = {
    'type': 'cross_entropy',
    'encoder': "bert",
}

trainer_params = {
    'shuffle': True,
    'num_workers': 0
}

batch_size = {
    4: 4, 8: 8, 16: 12
}

epochs = {
    4: {
        2: 15, 3: 30, 4: 45, 5: 60, 6: 75
    },
    8: {
        2: 30, 3: 40, 4: 50, 5: 70, 6: 100
    },
    16: {
        2: 15, 3: 25, 4: 35, 5: 45, 6: 55
    },
}

learning_rates = {
    4: 1e-3,
    8: 1e-3,
    16: 1e-3
}

training_params = {
    'encoder': "bert",
    'epochs': epochs,
    'warmupSteps': 100,
    'reduction': 'none',
    'learning_rate': learning_rates,
    'printValidationPlot': False,
    'printValidationLoss': True
}

for category in ["rating_dvd"]:
    training_params['category'] = category
    for episode in range(10):
        training_params['episode'] = episode
        for shot in [8]:
            training_params['shot'] = shot
            trainer_params['batch_size'] = batch_size[shot]
            training_encodings, training_labels, label_keys = get_labelled_LEOPARD_training_data(category, shot, episode)

            if training_encodings.shape[0] == 0:
                continue

            training_set = {}
            training_set['encodings'] = training_encodings
            training_set['labels'] = training_labels

            metaClassifier = "/scratch/users/k21036268/models/version_8230128/checkpoints/working" + ".ckpt"
            lineGenerator = LineGenerator(training_set, PROTOTYPE_META_MODEL)
            classifierTrainer = MetaClassifier(metaClassifier, lineGenerator, trainer_params, training_params, label_keys)
            classifierTrainer.trainPrototypes(training_params, trainer_params, training_set)
            lines = classifierTrainer.getLines()

            evaluator = Evaluator(label_keys, test_params)
            evaluator.evaluate(category, shot, episode)