%%writefile main_federated.py
import flwr as fl
from models import EVModel, HEVModel, PHEVModel
from vehicle_client import VehicleClient
from data_generator import generate_ev_data, generate_hev_data, generate_phev_data
from torch.utils.data import DataLoader

def client_fn(cid):
    if cid == "0":
        model = EVModel()
        ds = generate_ev_data()
        dl = DataLoader(ds, batch_size=16)
        return VehicleClient(model, dl, dl, "EVModel", is_classification=False)
    elif cid == "1":
        model = HEVModel()
        ds = generate_hev_data()
        dl = DataLoader(ds, batch_size=16)
        return VehicleClient(model, dl, dl, "HEVModel", is_classification=True)
    elif cid == "2":
        model = PHEVModel()
        ds = generate_phev_data()
        dl = DataLoader(ds, batch_size=16)
        return VehicleClient(model, dl, dl, "PHEVModel", is_classification=False)

# Run the federated simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=5),
)
