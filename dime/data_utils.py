import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset, TensorDataset
import pickle

class HEVDataset(Dataset):
    def __init__(self, data_dir, feature_list=None, cols_to_drop=None):
        # Load data.
        data = pd.read_csv("/content/drive/MyDrive/dataset/dataset.csv")
        print(data.columns)

        if cols_to_drop is not None:
            data = data.drop(columns=cols_to_drop)

        # Set features, inputs and outputs.
        if feature_list is not None:
            self.features = feature_list
        else:
            self.features = [f for f in data.columns if f not in ['faultNumber']]
        self.X = np.array(data.drop(['faultNumber'], axis=1)[self.features]).astype('float32')
        self.Y = np.array(data['faultNumber']).astype('int64')

        # Create dataset object.
        # Set input size and output size.
        self.input_size = self.X.shape[1]
        self.output_size = len(np.unique(self.Y))

        # Create dataset object.
        self.dataset = TensorDataset(torch.from_numpy(self.X), torch.from_numpy(self.Y))
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def get_group_matrix(features, feature_groups):
    # Add singleton groups.
    complete_groups = {}
    groups_union = []
    for key in feature_groups:
        groups_union += feature_groups[key]
        complete_groups[key] = feature_groups[key]
    for feature in features:
        if feature not in groups_union:
            complete_groups[feature] = [feature]

    # Create groups matrix.
    group_matrix = np.zeros((len(complete_groups), len(features)))
    for i, key in enumerate(complete_groups):
        inds = [features.index(feature) for feature in complete_groups[key]]
        group_matrix[i, inds] = 1

    return complete_groups, group_matrix


def get_groups_dict_mask(feature_groups, num_feature):
    group_start = list(feature_groups.keys())
    feature_groups_dict = {}
    num_group = 0
    i = 0
    while i < num_feature:
        feature_groups_dict[num_group] = []
        if i in group_start:
            for j in range(feature_groups[i]):
                feature_groups_dict[num_group].append(i+j)
            num_group += 1
            i += feature_groups[i]
        else:
            feature_groups_dict[num_group].append(i)
            num_group += 1
            i += 1
    feature_groups_mask = np.zeros((num_feature, len(feature_groups_dict)))
    for i in range(len(feature_groups_dict)):
        for j in feature_groups_dict[i]:
            feature_groups_mask[j, i] = 1
    return feature_groups_dict, feature_groups_mask


def get_xy(dataset):
    x, y = zip(*list(dataset))
    if isinstance(x[0], np.ndarray):
        return np.array(x), np.array(y)
    elif isinstance(x[0], torch.Tensor):
        if isinstance(y[0], (int, float)):
            return torch.stack(x), torch.tensor(y)
        else:
            return torch.stack(x), torch.stack(y)
    else:
        raise ValueError(f'not sure how to concatenate data type: {type(x[0])}')


def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0):
    # Shuffle sample indices.
    rng = np.random.default_rng(random_state)
    inds = np.arange(len(dataset))
    rng.shuffle(inds)

    # Assign indices to splits.
    n_val = int(val_portion * len(dataset))
    n_test = int(test_portion * len(dataset))
    test_inds = inds[:n_test]
    val_inds = inds[n_test:(n_test + n_val)]
    train_inds = inds[(n_test + n_val):]

    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)
    return train_dataset, val_dataset, test_dataset
