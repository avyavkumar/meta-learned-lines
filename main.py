#!/usr/bin/env python3
import os
import random
from argparse import ArgumentParser
from training_datasets.GLUEMetaDataset import GLUEMetaDataset
from samplers.FewShotEpisodeBatchSampler import FewShotEpisodeBatchSampler
import torch.utils.data as data
from samplers.FewShotValidationEpisodeBatchSampler import FewShotValidationEpisodeBatchSampler
from validation_datasets.ValidationDataset import ValidationDataset
from training.models.ProtoFOMAML import ProtoFOMAML
from training.trainer import train_model


def main(hyper_params):
    meta_dataset = GLUEMetaDataset(k=hyper_params.kShot, numTasks=hyper_params.numTasks)
    train_protomaml_sampler = FewShotEpisodeBatchSampler(meta_dataset, kShot=hyper_params.kShot,
                                                         batchSize=hyper_params.batchSize)
    train_protomaml_loader = data.DataLoader(
        meta_dataset,
        batch_sampler=train_protomaml_sampler,
        collate_fn=train_protomaml_sampler.getCollateFunction(),
        num_workers=8)
    validation_dataset = ValidationDataset()
    val_protomaml_sampler = FewShotValidationEpisodeBatchSampler(validation_dataset, kShot=hyper_params.kShot)
    val_protomaml_loader = data.DataLoader(
        validation_dataset,
        batch_sampler=val_protomaml_sampler,
        collate_fn=val_protomaml_sampler.getCollateFunction(),
        num_workers=4
    )
    # pick a randomised seed
    seed = random.randint(0, 10000)
    protomaml_model = train_model(
        ProtoFOMAML,
        seed=seed,
        train_loader=train_protomaml_loader,
        val_loader=val_protomaml_loader,
        outerLR=hyper_params.outerLR,
        innerLR=hyper_params.innerLR,
        outputLR=hyper_params.outputLR,
        steps=hyper_params.steps,
        batchSize=hyper_params.batchSize,
        warmupSteps=hyper_params.warmupSteps
    )
    return protomaml_model

# Example command - python main.py --outerLR 5e-4 --innerLR 1e-3 --outputLR 1e-2 --steps 5 --batchSize 4
# --warmupSteps 0 --kShot 4 --numTasks 10000

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-o', '--outerLR', type=float)
    parser.add_argument('-i', '--innerLR', type=float)
    parser.add_argument('-l', '--outputLR', type=float)
    parser.add_argument('-s', '--steps', type=int)
    parser.add_argument('-b', '--batchSize', type=int)
    parser.add_argument('-w', '--warmupSteps', type=int)
    parser.add_argument('-k', '--kShot', type=int)
    parser.add_argument('-n', '--numTasks', type=int)
    hyper_params = parser.parse_args()

    # TRAIN
    protomaml_model = main(hyper_params)
