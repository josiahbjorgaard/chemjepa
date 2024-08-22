from datasets import load_from_disk

import datasets
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch

class PairsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, source, mask = 0.0, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = datasets.load_from_disk(source).with_format('torch')
        self.transform = transform
        self.keys = {'pos':'coordinates1',
                     'pos2':'coordinates2',
                     'z':'atomic_numbers',
                     'label':'wb97x_dz.cm5_charges2',
                     'label2':'wb97x_dz.cm5_charges1'}
        self.mask =  mask
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = {k:sample[v] for k,v in self.keys.items()}
        if self.mask:
            sample['mask'] = torch.FloatTensor(sample['pos'].shape[0]).uniform_() < self.mask
        return Data(**sample)


def setup_data(dataset_path, split = 0.1, ds_frac=1.0, ds_seed=42):
    ## Dataset processing
    dataset = load_from_disk(dataset_path).with_format('torch')
    if ds_frac < 1.0:
        dataset = dataset.select(list(range(0,int(len(dataset)*ds_frac))))

    if split and split != 1.0:
        dataset = dataset.train_test_split(split, seed=ds_seed)
    return dataset
