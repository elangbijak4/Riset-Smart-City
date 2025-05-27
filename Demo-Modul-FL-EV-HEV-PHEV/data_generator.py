%%writefile data_generator.py
import torch
from torch.utils.data import TensorDataset

def generate_ev_data(n_samples=100):
    X = torch.rand(n_samples, 10)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + 0.1 * torch.randn(n_samples, 1)
    return TensorDataset(X, y)

def generate_hev_data(n_samples=100):
    X = torch.rand(n_samples, 12)
    y = (X[:, 0] + X[:, 1] > 1).long()  # Label 0 atau 1
    return TensorDataset(X, y)

def generate_phev_data(n_samples=100):
    X = torch.rand(n_samples, 15)
    y = (X[:, 0] * 0.7 + X[:, 1] * 0.3).unsqueeze(1)
    return TensorDataset(X, y)
