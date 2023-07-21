import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from transformers import logging

logging.set_verbosity_error()

MODEL_PATH = "models/meta_learned_model/"


def train_model(modelType, train_loader, val_loader, seed=42, **args):
    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        default_root_dir=os.path.join(MODEL_PATH, modelType.__name__),
        accelerator="gpu",
        devices="auto",
        max_epochs=200000,
        check_val_every_n_epoch=100,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="outer_loop_validation_loss"),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="outer_loop_validation_loss", patience=7, verbose=False, mode="min")
        ],
        enable_progress_bar=False,
    )
    trainer.logger._default_hp_metric = None

    # if a model exists, use that instead of training a new one
    existing_model = os.path.join(MODEL_PATH, modelType.__name__ + ".ckpt")
    if os.path.isfile(existing_model):
        print("Using model", existing_model)
        # Automatically loads the model with the saved hyperparameters
        model = modelType.load_from_checkpoint(existing_model)
    else:
        model = modelType(**args)
    pl.seed_everything(seed)
    trainer.fit(model, train_loader, val_loader)
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model
