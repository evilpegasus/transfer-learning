"""
Run training

Ming Fong
LBNL 2023
"""

import os
import data_utils
from absl import logging, app, flags
import numpy as np
import jax
import jax.numpy as jnp



flags.DEFINE_string("optimizer", "adam", "Optimizer to use.")
flags.DEFINE_integer("epochs", 400, "Number of epochs.")
flags.DEFINE_integer("eval_every", 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer("test_every", 500, 'Evaluation frequency (in steps).')
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("batch_size", 1024, "Batch size.")
flags.DEFINE_integer("seed", 8, "Random seed.")


FLAGS = flags.FLAGS



def main(unused_args):
  
  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))
  logging.info("rng_key: %s", rng_key)
  
  logging.info("Starting training")
  print("asdfasdfasdfasfasdfasdfasdf")
  logging.info("logging.info example")
  logging.warning("logging.warning example")
  logging.error("logging.error example")
  # logging.fatal("logging.fatal example")
  logging.debug("logging.debug example")




if __name__ == "__main__":
  app.run(main)
