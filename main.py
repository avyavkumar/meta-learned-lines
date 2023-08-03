#!/usr/bin/env python3
import os
import random
from argparse import ArgumentParser

from training.models.FOMAML import FOMAML
from training.models.ProtoNet import ProtoNet
from training.models.Reptile import Reptile
from training_datasets.GLUEMetaDataset import GLUEMetaDataset
from samplers.FewShotEpisodeBatchSampler import FewShotEpisodeBatchSampler
import torch.utils.data as data
from samplers.FewShotValidationEpisodeBatchSampler import FewShotValidationEpisodeBatchSampler
from validation_datasets.ValidationDataset import ValidationDataset
from training.models.ProtoFOMAML import ProtoFOMAML
from training.trainer import train_model


# Example command - python main.py --model ProtoNet --lengthTasks 10000 --outerLR 5e-5 --innerLR 1e-2 --steps 5 --batchSize 8
# --warmupSteps 10 --kShot 8 --kValShot 8 --numTasks 200000
def train_protonet(hyper_params):
    meta_dataset = GLUEMetaDataset(k=hyper_params.kShot, numTasks=hyper_params.numTasks,
                                   length=hyper_params.lengthTasks)
    train_protomaml_loader = data.DataLoader(
        meta_dataset,
        num_workers=2)
    validation_dataset = ValidationDataset()
    val_protonet_sampler = FewShotValidationEpisodeBatchSampler(validation_dataset, kShot=hyper_params.kValShot)
    val_protonet_loader = data.DataLoader(
        validation_dataset,
        batch_sampler=val_protonet_sampler,
        collate_fn=val_protonet_sampler.getCollateFunction(),
        num_workers=2
    )
    # pick a randomised seed
    seed = random.randint(0, 10000)
    protonet_model = train_model(
        ProtoNet,
        seed=seed,
        train_loader=train_protomaml_loader,
        val_loader=val_protonet_sampler,
        metaLearningRate=hyper_params.outerLR,
        prototypeLearningRate=hyper_params.innerLR,
        steps=hyper_params.steps,
        batchSize=hyper_params.batchSize,
        warmupSteps=hyper_params.warmupSteps
    )
    return protonet_model


# Example command - python main.py --model ProtoFOMAML --outerLR 5e-4 --innerLR 1e-3 --outputLR 1e-2 --steps 5 --batchSize 4
# --warmupSteps 0 --kShot 4 --kValShot 4 --numTasks 10000
def train_protomaml(hyper_params):
    meta_dataset = GLUEMetaDataset(k=hyper_params.kShot, numTasks=hyper_params.numTasks,
                                   length=hyper_params.lengthTasks)
    train_protomaml_sampler = FewShotEpisodeBatchSampler(meta_dataset, kShot=hyper_params.kShot,
                                                         batchSize=hyper_params.batchSize)
    train_protomaml_loader = data.DataLoader(
        meta_dataset,
        batch_sampler=train_protomaml_sampler,
        collate_fn=train_protomaml_sampler.getCollateFunction(),
        num_workers=2)
    validation_dataset = ValidationDataset()
    val_protomaml_sampler = FewShotValidationEpisodeBatchSampler(validation_dataset, kShot=hyper_params.kValShot)
    val_protomaml_loader = data.DataLoader(
        validation_dataset,
        batch_sampler=val_protomaml_sampler,
        collate_fn=val_protomaml_sampler.getCollateFunction(),
        num_workers=2
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


def train_fomaml(hyper_params):
    meta_dataset = GLUEMetaDataset(k=hyper_params.kShot, numTasks=hyper_params.numTasks,
                                   length=hyper_params.lengthTasks)
    train_protomaml_sampler = FewShotEpisodeBatchSampler(meta_dataset, kShot=hyper_params.kShot,
                                                         batchSize=hyper_params.batchSize)
    train_protomaml_loader = data.DataLoader(
        meta_dataset,
        batch_sampler=train_protomaml_sampler,
        collate_fn=train_protomaml_sampler.getCollateFunction(),
        num_workers=2)
    validation_dataset = ValidationDataset()
    val_protomaml_sampler = FewShotValidationEpisodeBatchSampler(validation_dataset, kShot=hyper_params.kValShot)
    val_protomaml_loader = data.DataLoader(
        validation_dataset,
        batch_sampler=val_protomaml_sampler,
        collate_fn=val_protomaml_sampler.getCollateFunction(),
        num_workers=2
    )
    # pick a randomised seed
    seed = random.randint(0, 10000)
    protomaml_model = train_model(
        FOMAML,
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


def train_reptile(hyper_params):
    meta_dataset = GLUEMetaDataset(k=hyper_params.kShot, numTasks=hyper_params.numTasks,
                                   length=hyper_params.lengthTasks)
    train_protomaml_sampler = FewShotEpisodeBatchSampler(meta_dataset, kShot=hyper_params.kShot,
                                                         batchSize=hyper_params.batchSize)
    train_protomaml_loader = data.DataLoader(
        meta_dataset,
        batch_sampler=train_protomaml_sampler,
        collate_fn=train_protomaml_sampler.getCollateFunction(),
        num_workers=2)
    validation_dataset = ValidationDataset()
    val_protomaml_sampler = FewShotValidationEpisodeBatchSampler(validation_dataset, kShot=hyper_params.kValShot)
    val_protomaml_loader = data.DataLoader(
        validation_dataset,
        batch_sampler=val_protomaml_sampler,
        collate_fn=val_protomaml_sampler.getCollateFunction(),
        num_workers=2
    )
    # pick a randomised seed
    seed = random.randint(0, 10000)
    reptile = train_model(
        Reptile,
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
    return reptile


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-o', '--outerLR', type=float)
    parser.add_argument('-i', '--innerLR', type=float)
    parser.add_argument('-l', '--outputLR', type=float)
    parser.add_argument('-s', '--steps', type=int)
    parser.add_argument('-b', '--batchSize', type=int)
    parser.add_argument('-w', '--warmupSteps', type=int)
    parser.add_argument('-k', '--kShot', type=int)
    parser.add_argument('-kv', '--kValShot', type=int)
    parser.add_argument('-n', '--numTasks', type=int)
    parser.add_argument('-f', '--lengthTasks', type=int)
    hyper_params = parser.parse_args()

    if hyper_params.model == "ProtoFOMAML":
        train_protomaml(hyper_params)
    elif hyper_params.model == "ProtoNet":
        train_protonet(hyper_params)
    elif hyper_params.model == "FOMAML":
        train_fomaml(hyper_params)
    elif hyper_params.model == "Reptile":
        train_reptile(hyper_params)
