import pytorch_lightning as pl

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

import nemo
import nemo.collections.asr as nemo_asr

from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import copy
import os
import datetime
import re


def train():
    # TODO: update matchboxnet_3x1x64_v1.yaml according to your labels
    MODEL_CONFIG = "matchboxnet_3x1x64_v1.yaml"

    config_path = f"config/{MODEL_CONFIG}"

    config = OmegaConf.load(config_path)

    # Preserve some useful parameters
    labels = config.model.labels
    sample_rate = config.sample_rate

    config.model.train_ds.manifest_filepath = "/manifest_files/train_manifest.jsonl"
    config.model.validation_ds.manifest_filepath = "/manifest_files/val_manifest.jsonl"
    config.model.test_ds.manifest_filepath = "/manifest_files/test_manifest.jsonl"


    save_path = os.path.join(os.getcwd(), "checkpoints/")
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=-1,
        mode='min',
        prefix='',
        period=1
    )

    time = str(datetime.date.today())
    run_name = f"speech_classification_training"
    wandb_logger = WandbLogger(name=run_name, project="SpeechClassification")

    # create trainer
    trainer = pl.Trainer(gpus=2, max_epochs=200, amp_level='O1', precision=16,
                         max_steps=None, num_nodes=1, accelerator="ddp",
                         accumulate_grad_batches=1, checkpoint_callback=checkpoint_callback, val_check_interval=1.0,
                         logger=wandb_logger, log_every_n_steps=100)

    # create model
    asr_model = nemo_asr.models.EncDecClassificationModel(cfg=config.model, trainer=trainer)

    # start training
    trainer.fit(asr_model)


if __name__ == "__main__":
    from eval import *
    is_train = True

    if is_train:
        train()
    else:
        setup_and_evaluate()







