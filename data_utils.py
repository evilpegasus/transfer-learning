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
      print("Working on", filepath)
      processed_data = preprocessing.constituent(old_file, 200)
      labels = old_file["labels"]
      print(f"Saving preprocessed data to {target_filepath}")
      dset = new_file.create_dataset("data", processed_data.shape)
      dset[:] = processed_data
      labels_dset = new_file.create_dataset("labels", labels.shape)
      labels_dset[:] = labels


class H5Dataset(Dataset):
  """
  Pytorch Dataset class for h5 files that loads in h5 files one at a time to save memory.
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
    self.opened_file_idx = None     # self.filepaths index that is opened
    self.opened_file = None         # h5 file that is opened

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
    self._open_h5_file(filepath_idx)
    
    labels = self.opened_file[self.LABEL_KEY][sample_idx]
    
    if self.preprocessed:
      data = self.opened_file[self.DATA_KEY][sample_idx]
    else:
      data = preprocessing.constituent({
        "fjet_clus_eta": self.opened_file["fjet_clus_eta"][sample_idx],
        "fjet_clus_phi": self.opened_file["fjet_clus_phi"][sample_idx],
        "fjet_clus_pt": self.opened_file["fjet_clus_pt"][sample_idx],
        "fjet_clus_E": self.opened_file["fjet_clus_E"][sample_idx],
      }, 200)

    # if self.transform:
    #   data = self.transform(data)

    data = np.asarray(data).flatten()
    labels = np.asarray(labels)
    return data, labels

  def _open_h5_file(self, filepath_idx: int):
    """Opens the h5 file at self.filepaths[filepath_idx]. If another file was open, close it. If the file is already open, do nothing"""
    if self.opened_file_idx == filepath_idx:
      return
    if self.opened_file_idx is not None:
      self._close_h5_file()
    filepath = self.filepaths[filepath_idx]
    self.opened_file = h5py.File(filepath, "r")
    self.opened_file_idx = filepath_idx
    
  def _close_h5_file(self):
    """Closes the currently opened h5 file. If no file is open, do nothing"""
    if self.opened_file_idx is None:
      return
    self.opened_file.close()
    self.opened_file_idx = None
    self.opened_file = None
  
  def get_opened_file(self) -> str:
    """Returns the name of the h5 files currently opened"""
    return self.filepaths[self.opened_file_idx]

  # NOTE this is too slow
  # def _load_h5_to_mem(self, file_idx):
  #   """Loads the data of a single h5 file into memory. If self.loaded_file_idx is already loaded, do nothing"""
  #   if self.loaded_file_idx == file_idx:
  #     return
  #   print("loading file", self.filepaths[file_idx])
  #   filepath = self.filepaths[file_idx]
  #   with h5py.File(filepath, "r") as file:
  #     self.loaded_data = file[self.DATA_KEY][:]
  #     self.loaded_labels = file[self.LABEL_KEY][:]
  #   self.loaded_file_idx = file_idx

  # def _get_loaded_file(self):
  #   """Returns the name of the h5 files currently loaded into memory"""
  #   return self.filepaths[self.loaded_file_idx]



class H5Dataset2(Dataset):
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
    
    # if self.preprocessed:
    #   data = self.opened_file[self.DATA_KEY][sample_idx]
    # else:
    #   data = preprocessing.constituent({
    #     "fjet_clus_eta": self.opened_file["fjet_clus_eta"][sample_idx],
    #     "fjet_clus_phi": self.opened_file["fjet_clus_phi"][sample_idx],
    #     "fjet_clus_pt": self.opened_file["fjet_clus_pt"][sample_idx],
    #     "fjet_clus_E": self.opened_file["fjet_clus_E"][sample_idx],
    #   }, 200)

    # if self.transform:
    #   data = self.transform(data)

    data = np.asarray(data).flatten()
    labels = np.asarray(labels)
    return data, labels



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
  preprocess_data([sys.argv[1]], sys.argv[2])