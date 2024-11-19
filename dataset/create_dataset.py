import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, IterableDataset
from torch.utils.data import WeightedRandomSampler
# import bisect
# import warnings
from typing import (
    Iterable,
    List,
    Optional,
    TypeVar,
)
# from operator import itemgetter

# from .raft import RAFTExhaustiveDataset#, RAFTDepthDataset
from .raft_online import SimpleDepthDataset
from .longterm import LongtermDataset
# from .gm_online import GMDepthDataset, GMExhaustiveDataset
# from .RGB_online import RGBDepthDataset
# from .raft_offline import OfflineDepthDataset
# from .raft_random import SimpleRandomDataset


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


dataset_dict = {
    'long': LongtermDataset,
    'simple': SimpleDepthDataset
}   

class StackDataset(Dataset[T_co]):
    r"""Dataset as a stack of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be stacked
    """
    datasets: List[Dataset[T_co]]

    def __init__(self, datasets: Iterable[Dataset], dataset_types:list) -> None:
        super(StackDataset, self).__init__()
        self.datasets = list(datasets)
        self.dataset_types = dataset_types
        assert len(self.datasets) > 0, 'datasets should not be an empty'
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "StackDataset does not support IterableDataset"

    # def increase_range(self):
    #     for dataset in self.datasets:
    #         dataset.increase_range()

    def __len__(self):
        return max(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        return {dataset_type: dataset[idx] for dataset, dataset_type in zip(self.datasets, self.dataset_types)}

def get_training_dataset(args, max_interval):
    dataset_types = args.dataset_types.split('+')
    weights = args.dataset_weights
    assert len(dataset_types) == len(weights), 'Number of dataset types should match number of weights'
    assert np.abs(np.sum(weights) - 1.) < 1e-6, 'Weights should sum to 1'
    train_datasets = []
    train_weights_samples = []
    for dataset_type, weight in zip(dataset_types, weights):
        train_dataset = dataset_dict[dataset_type](args, max_interval=max_interval)
        train_datasets.append(train_dataset)
        num_samples = len(train_dataset)
        weight_each_sample = weight / num_samples
        train_weights_samples.extend([weight_each_sample]*num_samples)

    train_dataset = StackDataset(train_datasets, dataset_types)
    train_weights = torch.from_numpy(np.array(train_weights_samples))
    sampler = WeightedRandomSampler(train_weights, len(train_weights))
    train_sampler = sampler

    return train_dataset, train_sampler


