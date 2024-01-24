"""
Run training

Ming Fong
LBNL 2023
"""

from typing import Sequence
import os
from absl import logging, app, flags
import numpy as np
import jax
import jax.numpy as jnp
import wandb
import optax
from flax.training import train_state, orbax_utils
import orbax.checkpoint
from sklearn.metrics import roc_auc_score
import models

import data_utils

flags.DEFINE_list("dnn_layers", [400, 400, 400, 400, 400, 1], "DNN layers.")
flags.DEFINE_enum("optimizer", "adam", ["adam"], "Optimizer to use.")
flags.DEFINE_integer("epochs", 400, "Number of epochs. "
                     "If resume_training, this is includes previously run epochs.")
flags.DEFINE_integer("eval_every", 1, "Evaluation frequency (in epochs).")
flags.DEFINE_integer("test_every", 1, "Evaluation frequency (in epochs).")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
flags.DEFINE_integer("batch_size", 1024, "Batch size.")
flags.DEFINE_enum("loss", "binary_crossentropy",["binary_crossentropy"], "Loss function.")
flags.DEFINE_integer("seed", 8, "Random seed.")
flags.DEFINE_integer("num_files", 1, "Number of files to use for training.")
flags.DEFINE_integer("max_train_rows", None,
                     "Maximum number of rows to use for training. If None, use all rows.")
flags.DEFINE_integer("max_val_rows", None,
                     "Maximum number of rows to use for validation. If None, use all rows.")
flags.DEFINE_enum("dataload_method", "all", ["single", "all"],
                  "Method for loading data. If single, load one batch at a time (slow, saves memory). "
                  "If all, load all data into memory (fast, high memory consumption).")
flags.DEFINE_string("train_dir", "/pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/",
                    "Directory of preprocessed training data.")
flags.DEFINE_string("test_dir", "/pscratch/sd/m/mingfong/transfer-learning/delphes_test_processed/",
                    "Directory of preprocessed testing data.")
flags.DEFINE_string("wandb_run_name", None,
                    "WandB run name. If None, makes a default name with hyperparameters.")
flags.DEFINE_string("wandb_project", "delphes_pretrain",
                    "WandB project name.")
flags.DEFINE_string("wandb_run_path", None,
                    "Previous WandB run path to load checkpoint. (i.e. mingfong/fullsim/15yzruts). "
                    "If None or if resume_training is False, create a new run.")
flags.DEFINE_bool("resume_training", False,
                  "Whether to resume training from checkpoint. Will start the epoch count from the previous run. "
                  "If True, must specify wandb_run_path. "
                  "If False, only loads weights if wandb_run_path is specified.")
flags.DEFINE_integer("checkpoint_interval", 1, "Checkpoint interval (in epochs).")

FLAGS = flags.FLAGS


def process_flags():
  """Process flags."""
  FLAGS.dnn_layers = [int(layer) for layer in FLAGS.dnn_layers]
  if FLAGS.wandb_run_path is None and FLAGS.resume_training:
    raise ValueError("Must specify wandb_run_path if resume_training is True.")

 
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
  logits = state.apply_fn(state.params, x).squeeze()
  loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=y))
  return loss, logits

def main(unused_args):
  process_flags()
  logging.warning(f"unsed_args: {unused_args}")

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))
  logging.info("rng_key: %s", rng_key)
  logging.info("Devices: %s", jax.devices())

  # Initialize data
  logging.info("Initializing data...")
  train_dir_preprocess = FLAGS.train_dir   # directory of preprocessed training data
  train_preprocess_file_names = os.listdir(train_dir_preprocess)
  train_preprocess_filepaths = [train_dir_preprocess + name for name in train_preprocess_file_names]

  if FLAGS.dataload_method == "single":
    raise NotImplementedError("single dataload_method not supported")
    DatasetClassToUse = data_utils.H5DatasetLoadSingle
  elif FLAGS.dataload_method == "all":
    DatasetClassToUse = data_utils.H5DatasetLoadAll
  else:
    raise ValueError(f"Unsupported dataload_method: {FLAGS.dataload_method}")
  train_dataset = DatasetClassToUse(train_preprocess_filepaths[0:FLAGS.num_files], max_rows=FLAGS.max_train_rows)
  train_dataloader = data_utils.JaxDataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)
  logging.info("Num train samples: %s", len(train_dataset))

  val_dataset = DatasetClassToUse(train_preprocess_filepaths[-1:], max_rows=FLAGS.max_val_rows, reverse_data=True)
  val_dataloader = data_utils.JaxDataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False)
  logging.info("Num val samples: %s", len(val_dataset))

  dummy_input = next(iter(train_dataloader))[0]
  logging.info("Input shape: %s", dummy_input.shape)

  # WandB setup
  config = {
  "seed": FLAGS.seed,
  "epochs": FLAGS.epochs,
  "batch_size": FLAGS.batch_size,
  "learning_rate": FLAGS.learning_rate,
  "optimizer": FLAGS.optimizer,
  "loss": FLAGS.loss,
  "train_samples": len(train_dataset),
  "test_samples": len(val_dataset),
  }
  wandb_run_name = FLAGS.wandb_run_name if FLAGS.wandb_run_name is not None else f"MLP rows={int(config['train_samples'] / 1000000)}M lr={config['learning_rate']} B={config['batch_size']} epochs={config['epochs']} dnn_layers={FLAGS.dnn_layers}"
  wandb_run = wandb.init(
    project=FLAGS.wandb_project,
    name=wandb_run_name,
    dir="/pscratch/sd/m/mingfong/transfer-learning/",
    config=config, resume="allow",
    id=FLAGS.wandb_run_path.split("/")[-1] if (FLAGS.wandb_run_path is not None and FLAGS.resume_training) else None,
  )
  # TODO tag wandb run with config params instead of dumping into name

  # Initialize model
  logging.info("Initializing model")
  model = models.MLP(features=FLAGS.dnn_layers)

  if FLAGS.optimizer == "adam":
    opt = optax.adam(FLAGS.learning_rate)
  else:
    raise ValueError(f"Unsupported optimizer: {FLAGS.optimizer}")
  state = init_train_state(rng_key, model, opt, dummy_input)
  
  logging.info("wandb.run.dir: %s", wandb.run.dir)
  # Checkpointer setup
  ckpt = {
    "step": 0,
    "state": state,
  }
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(ckpt)
  
  start_step = 1
  if FLAGS.wandb_run_path is not None:
    # restore checkpoint files from wandb previous run
    logging.info("Restoring weights from previous run")
    artifact = wandb.use_artifact(
      f"{FLAGS.wandb_run_path}-checkpoint:latest"
    )
    artifact_dir = artifact.download(wandb.run.dir + "/orbax_ckpt")

    # Restore the latest checkpoint
    raw_restored = orbax_checkpointer.restore(artifact_dir, item=ckpt)
    state = raw_restored["state"]
    if wandb.run.resumed:
      logging.info("Restoring step from previous run")
      start_step = raw_restored["step"] + 1
  else:
    logging.info("Training from scratch.")


  # Training loop
  logging.info("Starting training")
  for epoch in range(start_step, FLAGS.epochs + 1):
    best_val_loss = 1e9

    # Training
    train_datagen = iter(train_dataloader)
    train_batch_matrics = {
      "loss": [],
      "accuracy": [],
      "auc": [],
    }
    max_batch_step = len(train_dataloader) - 1
    for batch_step in range(len(train_dataloader)):
      batch = next(train_datagen)
      state, loss, logits = train_step(state, batch)
      accuracy = jnp.mean((logits > 0) == batch[1])
      auc = roc_auc_score(batch[1], logits)
      train_batch_matrics["loss"].append(loss)
      train_batch_matrics["accuracy"].append(accuracy)
      train_batch_matrics["auc"].append(auc)

      # batch level logging
      wandb.log({
        "batch/train_loss": loss,
        "batch/train_accuracy": accuracy,
        "batch/train_auc": auc,
        "batch/batch_step": batch_step + epoch * (max_batch_step + 1),
      }, commit=(max_batch_step != batch_step)) # don't commit on last batch, let epoch level logging commit

    # Validation
    if epoch % FLAGS.eval_every == 0:
      val_datagen = iter(val_dataloader)
      val_batch_matrics = {
        "loss": [],
        "accuracy": [],
        "auc": [],
      }
      for batch_step in range(len(val_dataloader)):
        batch = next(val_datagen)
        loss, logits = eval_step(state, batch)
        val_batch_matrics["loss"].append(loss)
        val_batch_matrics["accuracy"].append(jnp.mean((logits > 0) == batch[1]))
        val_batch_matrics["auc"].append(roc_auc_score(batch[1], logits))

    train_loss = np.mean(train_batch_matrics["loss"])
    train_acc = np.mean(train_batch_matrics["accuracy"])
    train_auc = np.mean(train_batch_matrics["auc"])
    val_loss = np.mean(val_batch_matrics["loss"])
    val_acc = np.mean(val_batch_matrics["accuracy"])
    val_auc = np.mean(val_batch_matrics["auc"])
    logging.info(f"Epoch {epoch}/{FLAGS.epochs}: train_loss: {train_loss:.4f}, train_accuracy: {train_acc:.4f}, train_auc: {train_auc:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}, val_auc: {val_auc:.4f}")
    # Log metrics
    wandb.log({
      "epoch/train_loss": np.mean(train_batch_matrics["loss"]),
      "epoch/train_accuracy": np.mean(train_batch_matrics["accuracy"]),
      "epoch/train_auc": np.mean(train_batch_matrics["auc"]),
      "epoch/val_loss": np.mean(val_batch_matrics["loss"]),
      "epoch/val_accuracy": np.mean(val_batch_matrics["accuracy"]),
      "epoch/val_auc": np.mean(val_batch_matrics["auc"]),
      "epoch/epoch": epoch,
    }, commit=True)

    # save checkpoint every checkpoint_interval epochs or last epoch
    if epoch % FLAGS.checkpoint_interval == 0 or epoch == FLAGS.epochs:
      ckpt["step"] = epoch
      ckpt["state"] = state
      checkpoint_dir = wandb.run.dir + "/orbax_ckpt"
      orbax_checkpointer.save(checkpoint_dir, ckpt, save_args=save_args, force=True)   # single save
      artifact = wandb.Artifact(f"{wandb.run.id}-checkpoint", type="checkpoint")
      artifact.add_file(checkpoint_dir + "/checkpoint")
      wandb.log_artifact(artifact, aliases=["latest", f"step_{epoch}"])

  wandb.finish()
  

if __name__ == "__main__":
  app.run(main)

# https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-and-Flax--VmlldzoyMzA4ODEy
