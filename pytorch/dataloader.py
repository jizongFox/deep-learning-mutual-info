import scipy.io as sio
import os
import numpy as np
import random
from pathlib2 import Path
from torch.utils.data import TensorDataset, DataLoader
import torch


# load data from file
def load_data(name, random_labels=False):
    """Load the data
    name - the name of the dataset
    random_labels - True if we want to return random labels to the dataset
    return object with data and labels"""
    C = type('type_C', (object,), {})
    data_sets = C()
    d = sio.loadmat(str(Path(__file__).parents[1] / (name + '.mat')))
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    data_sets = C()
    data_sets.data = F
    data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    return data_sets


def get_data_loaders():
    dataset = load_data("var_u")
    data = dataset.data
    labels = dataset.labels

    # Split data in test and train
    test = random.sample(range(4096), 1000)
    train = ([i for i in range(4096) if i not in test])

    train_data = data[train]
    train_labels = labels[train][:, 1]
    test_data = data[test]
    test_labels = labels[test][:, 1]

    train_set = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())
    test_set = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())
    train_loader = DataLoader(train_set, batch_size=5000)
    test_loader = DataLoader(test_set, batch_size=5000)
    return train_loader, test_loader
