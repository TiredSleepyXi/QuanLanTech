import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.io import loadmat

class LoadDataset_from_numpy_X6(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy_X6, self).__init__()

        # load files
        # X_train_1 = np.load(np_dataset[0])
        X_train = loadmat(np_dataset[0]+'Data')["data"]
        y_train = loadmat(np_dataset[0]+'Label')["label"]

        for np_file in np_dataset:
            X_train = np.vstack((X_train, loadmat(np_file+'Data')["data"]))
            y_train = np.append(y_train, loadmat(np_file+'Label')["label"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train).float()
        self.y_data = torch.from_numpy(y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class LoadDataset_from_numpy_x6(Dataset):
    def __init__(self,np_dataset):
        super(LoadDataset_from_numpy_x6, self).__init__()

        #load files
        X_train =loadmat(np_dataset[0]['data'])
        Y_train =loadmat(np_dataset[0]['label'])

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train,np.load(np_file)["x"]))
            Y_train = np.append(Y_train,np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(Y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




def data_generator_np(training_files, subject_files, batch_size):
    train_dataset = LoadDataset_from_numpy_X6(training_files)
    test_dataset = LoadDataset_from_numpy_X6(subject_files)

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts