import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

""" Load Data """
class MoviesDataset():
    def __init__(self):
        dataset = pd.read_csv('DatasetMovies.csv', sep=',')
        X = dataset.iloc[:, 1:-1]
        y = dataset['rating']
        self.class_names = np.unique(y)
        self.num_classes = len(np.unique(y))
        self.X = X.values
        self.y = LabelEncoder().fit_transform(y)
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
