%%writefile vehicle_client.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
from utils import save_model

class VehicleClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, model_name, is_classification=False):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.is_classification = is_classification
        self.loss_fn = nn.NLLLoss() if is_classification else nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.model_name = model_name

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss_fn(output, y if self.is_classification else y)
                loss.backward()
                self.optimizer.step()
        save_model(self.model, f"{self.model_name}.pt")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                output = self.model(X)
                loss += self.loss_fn(output, y if self.is_classification else y).item()
                if self.is_classification:
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(y.view_as(pred)).sum().item()
        if self.is_classification:
            accuracy = correct / len(self.test_loader.dataset)
            return loss, len(self.test_loader.dataset), {"accuracy": accuracy}
        else:
            return loss, len(self.test_loader.dataset), {}
