import torch
import os
from torch.utils.data import Dataset


# Dataset class
class EEGImageNetDataset_Spam(Dataset):
    __dirname__ = os.path.dirname(__file__)

    __FILE_TRAIN_LOC__ = os.path.join(__dirname__, '../../content/EEGImageNetDataset_Spam/eeg_5_95_std.pth')

    # Constructor
    def __init__(self, dev, subject=0, sample_min_idx=20, sample_max_idx=460, model_type='model10'):
        # Load EEG signals
        loaded = torch.load(self.__FILE_TRAIN_LOC__)
        self.dev = dev
        if subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == subject]
        else:
            self.data = loaded['dataset']

        self.labels = loaded["labels"]
        self.images = loaded["images"]

        self.time_low = sample_min_idx
        self.time_high = sample_max_idx

        self.model_type = model_type

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high, :]

        if self.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, self.time_high - self.time_low)
        # Get label
        label = self.data[i]["label"]
        label_tensor = torch.tensor(label)
        pass
        # Return
        return eeg.to(self.dev), label_tensor.to(self.dev)

    @staticmethod
    def get_name():
        return "EEGImageNetDataset_Spam"


# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label
