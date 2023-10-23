"""
Train a fully connect DNN

Ming Fong 2023
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

from data_utils import preprocess_data, H5Dataset

