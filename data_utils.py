"""
Data loading utils for delphes and fullsim datasets.

Ming Fong
LBNL 2023
"""
import sys
import os
from typing import List
import h5py
from torch.utils.data import Dataset, DataLoader, default_collate
from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
import preprocessing
from tqdm import tqdm
from pathlib import Path
from absl import logging


def preprocess_data(train_filepaths: List[str], target_dir: str, force: bool=False, low_memory: bool=True, num_constituents: int=200):
  """Preprocesses h5 data from train_filepaths using the preprocess.py script and stores it as h5 files in target_dir. Use absolute paths"""
  # make the path if it got purged from pscratch
  Path(target_dir).mkdir(parents=True, exist_ok=True)

  target_dir_filepaths = [target_dir + x for x in os.listdir(target_dir)]
  for filepath in tqdm(train_filepaths):
    filename = filepath.split("/")[-1]
    print(f"Starting preprocessing on {filepath}")
    target_filepath = f"{target_dir}preprocessed_{filename}"
    # check if this file was already preprocessed
    if not force and target_filepath in target_dir_filepaths:
      print(f"{target_filepath} is already in target_dir, skipping this file")
      continue
    with h5py.File(filepath, 'r') as original_file, h5py.File(target_filepath, "w") as new_file:
      print("Working on", filepath)
      if low_memory:
        chunk_size = 10000
        print(f"Low memory mode, processing {chunk_size} samples at a time")
        num_samples = len(original_file["labels"])
        dset_shape = (num_samples, num_constituents, 7)     # 200 constituents, 7 features
        dset = new_file.create_dataset("data", dset_shape)
        labels_dset = new_file.create_dataset("labels", num_samples)
        for i in tqdm(range(0, num_samples, chunk_size)):
          data_dict_chunk = {
            "fjet_clus_eta": original_file["fjet_clus_eta"][i:i+chunk_size],
            "fjet_clus_phi": original_file["fjet_clus_phi"][i:i+chunk_size],
            "fjet_clus_pt": original_file["fjet_clus_pt"][i:i+chunk_size],
            "fjet_clus_E": original_file["fjet_clus_E"][i:i+chunk_size],
          }
          processed_data_chunk = preprocessing.constituent(data_dict_chunk, num_constituents)
          labels_chunk = original_file["labels"][i:i+chunk_size]
          dset[i:i+chunk_size] = processed_data_chunk
          labels_dset[i:i+chunk_size] = labels_chunk
      else:
        processed_data = preprocessing.constituent(original_file, num_constituents)
        labels = original_file["labels"]
        print(f"Saving preprocessed data to {target_filepath}")
        dset = new_file.create_dataset("data", processed_data.shape)
        dset[:] = processed_data
        labels_dset = new_file.create_dataset("labels", labels.shape)
        labels_dset[:] = labels


class H5DatasetLoadSingle(Dataset):
  """
  Version of H5Dataset that opens and closes files for each sample.
  """
  def __init__(
    self,
    filepaths: List[str],
    transform=None,
    preprocessed:bool=True,
    data_key:str="data",
    label_key:str="labels",
    ):
    """Initialize the dataset object

    Args:
        filepaths (List[str]): List of full Linux filepaths to h5 files
        transform (Callable[np.ndarray], optional): Optional function that is applied to all samples. Defaults to None.
        preprocessed (bool, optional): If the data was already preprocessed set this to True. Defaults to True.
    """
    self.filepaths = filepaths
    self.transform = transform
    self.preprocessed = preprocessed
    self.FEATURE_KEYS = ['fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_pt', 'fjet_clus_E']       # original unprocessed feature keys
    self.DATA_KEY = data_key
    self.LABEL_KEY = label_key
    self.sample_indices = []        # Store a tuple of (filepath_idx, sample_idx) for each sample in the dataset

    for filepath_idx, file_path in enumerate(filepaths):
      with h5py.File(file_path, "r") as file:
        num_samples = len(file[self.LABEL_KEY])
        indices = list(range(num_samples))
        self.sample_indices.extend([(filepath_idx, idx) for idx in indices])

  def __len__(self):
    return len(self.sample_indices)

  def __getitem__(self, idx):
    """Returns a single sample from the dataset. Opens the h5 file containing the sample if it is not already open"""
    filepath_idx, sample_idx = self.sample_indices[idx]
    
    with h5py.File(self.filepaths[filepath_idx], "r") as file:
      labels = file[self.LABEL_KEY][sample_idx]
      data = file[self.DATA_KEY][sample_idx]

    # if self.transform:
    #   data = self.transform(data)

    data = np.ravel(np.asarray(data, dtype=jnp.float32))
    labels = np.asarray(labels)
    return data, labels


class H5DatasetLoadSingle2(Dataset):
  """
  Version of H5Dataset that opens all the h5 files and leaves them open.
  """
  def __init__(
    self,
    filepaths: List[str],
    # transform=None,
    data_key:str="data",
    label_key:str="labels",
    ):
    """Initialize the dataset object

    Args:
        filepaths (List[str]): List of full Linux filepaths to h5 files
        transform (Callable[np.ndarray], optional): Optional function that is applied to all samples. Defaults to None.
        preprocessed (bool, optional): If the data was already preprocessed set this to True. Defaults to True.
    """
    self.filepaths = filepaths
    # self.transform = transform
    self.DATA_KEY = data_key
    self.LABEL_KEY = label_key
    self.sample_indices = []        # Store a tuple of (filepath_idx, sample_idx) for each sample in the dataset
    self.opened_files = [None] * len(self.filepaths)          # list of opened h5 files

    for filepath_idx, file_path in enumerate(filepaths):
      self.opened_files[filepath_idx] = h5py.File(file_path, "r")
      num_samples = len(self.opened_files[filepath_idx][self.LABEL_KEY])
      indices = list(range(num_samples))
      self.sample_indices.extend([(filepath_idx, idx) for idx in indices])

  def __len__(self):
    return len(self.sample_indices)

  def __getitem__(self, idx):
    """Returns a single sample from the dataset. Opens the h5 file containing the sample if it is not already open"""
    filepath_idx, sample_idx = self.sample_indices[idx]
    
    labels = self.opened_files[filepath_idx][self.LABEL_KEY][sample_idx]
    data = self.opened_files[filepath_idx][self.DATA_KEY][sample_idx]

    # if self.transform:
    #   data = self.transform(data)

    data = np.ravel(np.asarray(data, dtype=jnp.float32))
    labels = np.asarray(labels)
    return data, labels


class H5DatasetLoadAll(Dataset):
  """
  Version of H5Dataset that loads all data into memory at initialization. Be careful of OOM errors. Much faster than other versions.
  """
  def __init__(
    self,
    filepaths: List[str],
    max_rows: int=None,
    reverse_data: bool=False,
    # transform=None,
    data_key:str="data",
    label_key:str="labels",
    ):
    """Initialize the dataset object

    Args:
        filepaths (List[str]): List of full Linux filepaths to h5 files
        max_rows (int, optional): Maximum number of rows to load from each file. Defaults to unlimited.
        reverse_data (bool, optional): If True, reverse the order of the data. Defaults to False.
        transform (Callable[np.ndarray], optional): Optional function that is applied to all samples. Defaults to None.
    """
    self.filepaths = filepaths
    # self.transform = transform
    self.DATA_KEY = data_key
    self.LABEL_KEY = label_key
    self.data = None
    self.labels = None
    self.length = 0
    self.max_rows = max_rows    # total max number of rows to load
    self.reverse_data = reverse_data
    
    if self.reverse_data:
      filepaths = filepaths[::-1]

    # first pass to get number of samples and array shapes
    data_shape = None
    label_shape = None
    for filepath in filepaths:
      with h5py.File(filepath, "r") as file:
        self.length += len(file[self.LABEL_KEY])
        # get data and label shape of one sample if not known
        if data_shape is None:
          data_shape = file[self.DATA_KEY][0].shape
        if label_shape is None:
          label_shape = file[self.LABEL_KEY][0].shape

    # read in all data and labels
    if self.max_rows is not None:
      self.length = min(self.length, self.max_rows)

    self.data = np.zeros((self.length, *data_shape), dtype=np.float32)
    self.labels = np.zeros((self.length, *label_shape), dtype=np.float32)
    logging.info(f"Created data array with shape {(self.length, *data_shape)}")
    logging.info(f"Created labels array with shape {(self.length, *label_shape)}")
    curr_idx = 0
    for filepath in tqdm(filepaths):
      with h5py.File(filepath, "r") as file:
        curr_len = len(file[self.LABEL_KEY])
        remaining_len = self.length - curr_idx
        rows_to_copy = min(curr_len, remaining_len)
        if self.reverse_data:
          self.data[curr_idx:curr_idx + rows_to_copy] = file[self.DATA_KEY][-rows_to_copy:]
          self.labels[curr_idx:curr_idx + rows_to_copy] = file[self.LABEL_KEY][-rows_to_copy:]
        else:
          self.data[curr_idx:curr_idx + rows_to_copy] = file[self.DATA_KEY][:rows_to_copy]
          self.labels[curr_idx:curr_idx + rows_to_copy] = file[self.LABEL_KEY][:rows_to_copy]
      curr_idx += rows_to_copy
      if curr_idx >= self.length:
        break

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    """Returns a single sample from the dataset. Opens the h5 file containing the sample if it is not already open"""    
    labels = self.labels[idx]
    data = self.data[idx]

    # if self.transform:
    #   data = self.transform(data)

    data = np.ravel(np.asarray(data, dtype=jnp.float32))
    labels = np.asarray(labels)
    return data, labels


def numpy_collate(batch):
  """Helper function for Dataloader to convert arrays to numpy."""
  return tree_map(np.asarray, default_collate(batch))


class JaxDataLoader(DataLoader):
  """Wrapper for pytorch Dataloader that converts to numpy for jax compatability."""
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


if __name__ == "__main__":
  """
  If run directly, preprocess the data and save it to the target_dir
  args[0] = filepath of the original data h5 file
  args[1] = path of target directory to save preprocessed data to
  """
  assert len(sys.argv) == 3, f"wrong number of arguments (should be 2, was {len(sys.argv) - 1})"
  print("Original data filepath:", sys.argv[1])
  print("Target directory:", sys.argv[2])
  
  # ml4hep filepath args:
  # /global/ml4hep/spss/mfong/transfer_learning/fullsim_test/test.h5
  # /global/ml4hep/spss/mfong/transfer_learning/fullsim_test_processed/
  # nohup python data_utils.py /global/ml4hep/spss/mfong/transfer_learning/fullsim_train/train.h5 /global/ml4hep/spss/mfong/transfer_learning/fullsim_train_processed/ > data_util_train.out &
  
  # nersc commond
  # nohup python3 data_utils.py /global/cfs/projectdirs/m3246/mingfong/transfer-learning/delphes_train_set/train_0.h5 /pscratch/sd/m/mingfong/transfer-learning/delphes_train_processed/ > data_util_train_0.out &
  preprocess_data([sys.argv[1]], sys.argv[2])
