import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KERAS_BACKEND"] = "jax"

from typing import List
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import preprocessing
from tqdm import tqdm
import keras_core as keras
from keras_core import layers
from keras_core import ops
import matplotlib.pyplot as plt

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from data_utils import preprocess_data, H5Dataset, H5DatasetLoadSingle

train_dir_preprocess = "/pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/"   # directory of preprocessed training data
train_preprocess_file_names = os.listdir(train_dir_preprocess)
train_preprocess_filepaths = [train_dir_preprocess + name for name in train_preprocess_file_names]

train_dataset = H5DatasetLoadSingle(train_preprocess_filepaths[0:4], transform=None)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
print("Num train samples:", len(train_dataset))

test_dataset = H5DatasetLoadSingle(train_preprocess_filepaths[5:6], transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
print("Num test samples", len(test_dataset))

print(next(iter(train_dataloader))[0].shape)

config = {
  "epochs": 400,
  "batch_size": 1024,
  "learning_rate": 0.001,
  "optimizer": "adam",
  "loss": "binary_crossentropy",
  "train_samples": len(train_dataset),
  "test_samples": len(test_dataset),
}
wandb_run = wandb.init(
  project="delphes_pretrain",
  name=f"MLP_delphes",
  config=config, reinit=True
)

model = keras.Sequential([
  keras.Input(shape=(200 * 7,)),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(8, activation='relu'),
  layers.Dense(1, activation='sigmoid'),
])
model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=["accuracy", "AUC"])

train_history = model.fit(
  train_dataloader,
  validation_data=test_dataloader,
  batch_size=config["batch_size"],
  epochs=config["epochs"],
  callbacks=[
    WandbMetricsLogger(log_freq="batch"),
    WandbModelCheckpoint("models", save_freq="epoch"),
  ],
)

wandb.finish()
