%%writefile utils.py
import torch
import os

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
