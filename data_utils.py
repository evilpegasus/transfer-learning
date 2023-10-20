import sys
import os
from typing import List
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import preprocessing
from tqdm import tqdm

def preprocess_data(train_filepaths: List[str], target_dir: str, force=False):
  """Preprocesses h5 data from train_filepaths using the preprocess.py script and stores it as h5 files in target_dir. Use absolute paths"""
  target_dir_filepaths = [target_dir + x for x in os.listdir(target_dir)]
  for filepath in tqdm(train_filepaths):
    filename = filepath.split("/")[-1]
    print(f"Starting preprocessing on {filepath}")
    target_filepath = f"{target_dir}preprocessed_{filename}"
    # check if this file was already preprocessed
    if not force and target_filepath in target_dir_filepaths:
      print(f"{target_filepath} is already in target_dir, skipping this file")
      continue
    with h5py.File(filepath, 'r') as old_file, h5py.File(target_filepath, "w") as new_file:
      processed_data = preprocessing.constituent(old_file, 200)
      labels = old_file["labels"]
      print(f"Saving preprocessed data to {target_filepath}")
      dset = new_file.create_dataset("data", processed_data.shape)
      dset[:] = processed_data
      labels_dset = new_file.create_dataset("labels", labels.shape)
      labels_dset[:] = labels


class H5Dataset(Dataset):
  """Pytorch Dataset class for h5 files"""
  def __init__(self, filepaths: List[str], transform=None):
    self.filepaths = filepaths
    self.transform = transform
    self.sample_indices = []  # Store the indices of samples within each file
    # self.FEATURE_KEYS = ['fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_pt', 'fjet_clus_E']
    self.DATA_KEY = "data"
    self.LABEL_KEY = "labels"
    self.loaded_file_idx = None     # self.filepaths index that is currently loaded into memory
    self.loaded_data = None
    self.loaded_labels = None

    for filepath_idx, file_path in enumerate(filepaths):
      with h5py.File(file_path, "r") as file:
        num_samples = len(file[self.LABEL_KEY])
        indices = list(range(num_samples))
        self.sample_indices.extend([(filepath_idx, idx) for idx in indices])

  def __len__(self):
    return len(self.sample_indices)

  def __getitem__(self, idx):
    filepath_idx, sample_idx = self.sample_indices[idx]
    self.load_file(filepath_idx)
    data = self.loaded_data[sample_idx]
    labels = self.loaded_labels[sample_idx]
    
    # filepath = self.filepaths[filepath_idx]
    # with h5py.File(filepath, 'r') as file:
    #   data = file[self.DATA_KEY][sample_idx]  # Load a single sample
    #   labels = file[self.LABEL_KEY][sample_idx]

    # if self.transform:
    #   data = self.transform(data)
    data = np.asarray(data).flatten()
    labels = np.asarray(labels)
    return data, labels

  def load_file(self, file_idx):
    """Loads the data of a single h5 file into memory. If self.loaded_file_idx is already loaded, do nothing"""
    if self.loaded_file_idx == file_idx:
      return
    print("loading file", self.filepaths[file_idx])
    filepath = self.filepaths[file_idx]
    with h5py.File(filepath, "r") as file:
      self.loaded_data = file[self.DATA_KEY][:]
      self.loaded_labels = file[self.LABEL_KEY][:]
    self.loaded_file_idx = file_idx
    
  def get_loaded_file(self):
    """Returns the name of the h5 files currently loaded into memory"""
    return self.filepaths[self.loaded_file_idx]
