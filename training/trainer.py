import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from transformers import logging
logging.set_verbosity_error()

MODEL_PATH = "models/meta_learned_model/"

def train_model(modelType, train_loader, val_loader, seed=42, **args):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(MODEL_PATH, modelType.__name__),
        accelerator="auto",
        devices=1,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="outer_loop_validation_accuracy"), LearningRateMonitor("epoch"),
            EarlyStopping(monitor="outer_loop_validation_accuracy", min_delta=0.01, patience=5, verbose=False, mode="max")
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
        pl.seed_everything(seed)
        model = modelType(**args)
        trainer.fit(model, train_loader, val_loader)
        model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    return model
