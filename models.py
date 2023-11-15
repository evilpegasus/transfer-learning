import jax
from typing import Any, Callable, Sequence
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn

class MLP(nn.Module):
  """Simple MLP."""

  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x
