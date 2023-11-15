"""
Run training

Ming Fong
LBNL 2023
"""

from typing import Sequence
import os
import data_utils
from absl import logging, app, flags
import numpy as np
import jax
import jax.numpy as jnp
import wandb
from tqdm import tqdm
import optax
from flax.training import train_state
from sklearn.metrics import roc_auc_score
import models


flags.DEFINE_string("optimizer", "adam", "Optimizer to use.")
flags.DEFINE_integer("epochs", 400, "Number of epochs.")
flags.DEFINE_integer("eval_every", 1, 'Evaluation frequency (in steps).')
# flags.DEFINE_integer("test_every", 500, 'Evaluation frequency (in steps).')
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("batch_size", 1024, "Batch size.")
flags.DEFINE_string("loss", "binary_crossentropy", "Loss function.")
flags.DEFINE_integer("seed", 8, "Random seed.")
flags.DEFINE_integer("num_files", 1, "Number of files to use for training.")

FLAGS = flags.FLAGS


def init_train_state(rng_key, model, optimizer, batch):
  """Initialize training state."""
  params = model.init(rng_key, batch)
  return train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=optimizer)


@jax.jit
def train_step(
  state: train_state.TrainState,
  batch: Sequence[jnp.ndarray],
  ):
  """Perform a single training step."""
  x, y = batch
  def loss_fn(params):
    logits = state.apply_fn(params, x).squeeze()
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=y))
    # print(logits.shape)
    # print(y.shape)
    # print(loss)
    # print(optax.sigmoid_binary_cross_entropy(logits=logits, labels=y).shape)
    # raise ValueError()
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grad = grad_fn(state.params)
  state = state.apply_gradients(grads=grad)
  return state, loss, logits


@jax.jit
def eval_step(
  state: train_state.TrainState,
  batch: jnp.ndarray,
  ):
  """Perform a single evaluation step."""
  x, y = batch
  logits = state.apply_fn(state.params, x)
  loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=y))
  return loss, logits

def main(unused_args):
  logging.warning(f"unsed_args: {unused_args}")

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))
  logging.info("rng_key: %s", rng_key)
  logging.info("Devices: %s", jax.devices())
    
  # Initialize data
  logging.info("Initializing data...")
  train_dir_preprocess = "/pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/"   # directory of preprocessed training data TODO put this in config
  train_preprocess_file_names = os.listdir(train_dir_preprocess)
  train_preprocess_filepaths = [train_dir_preprocess + name for name in train_preprocess_file_names]

  train_dataset = data_utils.H5Dataset4(train_preprocess_filepaths[0:FLAGS.num_files])      # pick h5Dataset class 1-4 for various loading methods (see data_utils.py)
  train_dataloader = data_utils.JaxDataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)
  logging.info("Num train samples: %s", len(train_dataset))

  val_dataset = data_utils.H5Dataset4(train_preprocess_filepaths[-2:-1])
  val_dataloader = data_utils.JaxDataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False)
  logging.info("Num val samples %s", len(val_dataset))

  dummy_input = next(iter(train_dataloader))[0]
  logging.info("Input shape: %s", dummy_input.shape)

  # WandB setup
  config = {
  "epochs": FLAGS.epochs,
  "batch_size": FLAGS.batch_size,
  "learning_rate": FLAGS.learning_rate,
  "optimizer": FLAGS.optimizer,
  "loss": "binary_crossentropy",
  "train_samples": len(train_dataset),
  "test_samples": len(val_dataset),
  }
  wandb_run = wandb.init(
    project="delphes_pretrain",
    name=f"MLP_delphes",
    config=config, reinit=True
  )
  
  # Initialize model
  logging.info("Initializing model")
  model = models.MLP(features=[128, 64, 1])
  params = model.init(rng_key, dummy_input)
  logging.info(jax.tree_map(lambda x: x.shape, params))

  if FLAGS.optimizer == "adam":
    opt = optax.adam(FLAGS.learning_rate)
  else:
    raise ValueError(f"Unsupported optimizer: {FLAGS.optimizer}")
  state = init_train_state(rng_key, model, opt, dummy_input)
  
  # Training loop
  logging.info("Starting training")
  for epoch in range(FLAGS.epochs):
    print(f"Epoch {epoch}/{FLAGS.epochs}")
    best_val_loss = 1e9
    
    # Training
    train_datagen = iter(train_dataloader)
    train_batch_matrics = {
      "loss": [],
      "accuracy": [],
      "auc": [],
    }
    pbar = tqdm(range(len(train_dataloader)))
    for batch_index in pbar:
      batch = next(train_datagen)
      state, loss, logits = train_step(state, batch)
      accuracy = jnp.mean((logits > 0) == batch[1])
      auc = roc_auc_score(batch[1], logits)
      train_batch_matrics["loss"].append(loss)
      train_batch_matrics["accuracy"].append(accuracy)
      train_batch_matrics["auc"].append(auc)
      
      pbar.set_description(f"loss: {loss:.4f}, accuracy: {accuracy:.4f}, auc: {auc:.4f}")
      # batch level logging
      wandb.log({
        "batch/train_loss": loss,
        "batch/train_accuracy": accuracy,
        "batch/train_auc": auc,
      }, commit=True)
    
    # Validation
    if epoch % FLAGS.eval_every == 0:
      val_datagen = iter(val_dataloader)
      val_batch_matrics = {
        "loss": [],
        "accuracy": [],
        "auc": [],
      }
      for batch_index in range(len(val_dataloader)):
        batch = next(val_datagen)
        loss, logits = eval_step(state, batch)
        val_batch_matrics["loss"].append(loss)
        val_batch_matrics["accuracy"].append(jnp.mean((logits > 0) == batch[1]))
        val_batch_matrics["auc"].append(roc_auc_score(batch[1], logits))
    
    # Log metrics
    wandb.log({
      "train_loss": np.mean(train_batch_matrics["loss"]),
      "train_accuracy": np.mean(train_batch_matrics["accuracy"]),
      "train_auc": np.mean(train_batch_matrics["auc"]),
      "val_loss": np.mean(val_batch_matrics["loss"]),
      "val_accuracy": np.mean(val_batch_matrics["accuracy"]),
      "val_auc": np.mean(val_batch_matrics["auc"]),
    }, step=epoch, commit=True)

  wandb.finish()
  

if __name__ == "__main__":
  app.run(main)


# https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-and-Flax--VmlldzoyMzA4ODEy