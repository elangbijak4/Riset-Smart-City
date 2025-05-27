%%writefile models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EVModel(nn.Module):
    def __init__(self):
        super(EVModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Regress: konsumsi energi

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class HEVModel(nn.Module):
    def __init__(self):
        super(HEVModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)  # Classify: ICE vs Motor

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

class PHEVModel(nn.Module):
    def __init__(self):
        super(PHEVModel, self).__init__()
        self.fc1 = nn.Linear(15, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Regress: EV mode ratio

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
