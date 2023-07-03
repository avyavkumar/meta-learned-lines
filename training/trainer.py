import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

MODEL_PATH = "models/meta_learned_model/"

# adapted from https://lightning.ai/docs/pytorch/latest/notebooks/course_UvA-DL/12-meta-learning.html
def train_model(modelType, train_loader, val_loader, seed=42, **args):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(MODEL_PATH, modelType.__name__),
        accelerator="auto",
        devices=1,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
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
        trainer.fit(model, train_loader)
        model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    return model
