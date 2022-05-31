#!/usr/bin/env python
# coding: utf-8
import os, sys

if "adroit" in os.uname()[1]:
    CUSTOM_MODULE_PATH = "/home/jylin/thesis/modules"
else:
    CUSTOM_MODULE_PATH = "/System/Volumes/Data/Users/jesselin/Dropbox/src/thesis/modules"
sys.path.append(CUSTOM_MODULE_PATH)

# custom libraries
from entropy import get_entropy
from my_funcs import *

# usual libraries
import glob
import shutil
from collections import defaultdict
import argparse
import functools
import time
from tqdm import trange, tqdm
from datetime import datetime

# scientific libraries
import numpy as np

# ML libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

INTERACTION_Kc = np.log(1 + np.sqrt(2)) / 2


def make_dataloader(dataset: SpinSequenceDataset, num_samples: int, batch_size: int = 10, num_workers: int = 0):
    """ Takes dataset and returns the DataLoader object """
    div, mod = divmod(len(dataset), num_samples)

    # make list of the split proportions
    split_list = [num_samples for x in range(div)]
    split_list.append(mod)

    dataset_after_split = random_split(dataset, split_list)
    train_loader = DataLoader(dataset_after_split[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader


def get_temperature(f):
    f, _ = os.path.splitext(f)
    temperature = f[f.index("T=")+2:]
    return temperature


def check_for_checkpoint(log_dir, hidden_size):
    version_dir = os.path.join(log_dir, f"hidden_size={hidden_size}/version_0")
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        checkpoints = sorted(os.listdir(ckpt_dir))
        if len(checkpoints) == 0:
            shutil.rmtree(version_dir)
            print("Removing empty dir:", version_dir)
            return None
        else:
            return os.path.join(ckpt_dir, checkpoints[-1])
    else:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--res_dir", type=str)
    parser.add_argument("--max_hidden_size", type=int)
    parser.add_argument("--task_id", type=int)

    parsed_args_namespace = parser.parse_args()
    DATA_DIR = parsed_args_namespace.data_dir
    RES_DIR = parsed_args_namespace.res_dir
    MAX_HIDDEN_SIZE = parsed_args_namespace.max_hidden_size
    task_id = parsed_args_namespace.task_id

    res_dir = os.path.join(RES_DIR, "2D/compiled extra tensors")
    search_term = "compiled_extra*"
    raw_file_list = glob.glob(os.path.join(res_dir, search_term))
    data_dir = os.path.join(DATA_DIR, "2D")

    # divide hidden_size up between 4 parallel jobs
    hidden_size_iterator_full = np.arange(MAX_HIDDEN_SIZE) + 1
    hidden_size_iterator_full = np.array_split(hidden_size_iterator_full, 4)
    hidden_size_iterator = hidden_size_iterator_full[task_id]

    for f in raw_file_list:
        temperature_str = get_temperature(f)
        kfrac_str = str(round(1/float(temperature_str)/INTERACTION_Kc, 2))
        log_dir = os.path.join(data_dir, f"adroit_kfrac={kfrac_str}")

        dataset = SpinSequenceDataset(f, interaction_K=float(kfrac_str)*INTERACTION_Kc)
        dataloader = make_dataloader(dataset, num_samples=100, num_workers=32)
        for hidden_size in hidden_size_iterator:
            checkpoint = check_for_checkpoint(log_dir, hidden_size)
            print("Resuming checkpoint: ", checkpoint)
            print("\n")
            print("Starting...")
            print("hidden_size=", hidden_size)
            print("kfrac=", kfrac_str)
            print("\n")
            model = IsingRNN_simple(hidden_size=hidden_size, num_layers=1, nonlinearity="tanh", bias_on=False)
            logger = TensorBoardLogger(save_dir=log_dir, name=f"hidden_size={hidden_size}", version=0)
            early_stop = EarlyStopping(monitor="train_loss",
                                       check_on_train_epoch_end=True,
                                       min_delta=0.001,
                                       patience=10)
            trainer = pl.Trainer(logger=logger,
                                 max_epochs=100,
                                 resume_from_checkpoint=checkpoint,
                                 callbacks=[early_stop],
                                 gpus=1)
            trainer.fit(model, dataloader)
    return


if __name__ == "__main__":
    main()
