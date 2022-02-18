import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from dataset_utils import getCIFAR10, do_fl_partitioning, get_dataloader

from torch.profiler import profile, record_function, ProfilerActivity
import os
import pickle as pkl


# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# borrowed from Pytorch quickstart example
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()

    print("start profiling...")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(epochs):
           for images, labels in trainloader:
               images, labels = images.to(device), labels.to(device)
               optimizer.zero_grad()
               loss = criterion(net(images), labels)
               loss.backward()
               optimizer.step()
    list1 = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=0)
    print("end profiling")

    import re
    find_float = lambda x: re.search("\d+(\.\d+)?s", x).group()
    cpu_time = float(find_float(str(list1))[:-1])
    print("cpu_time", cpu_time)
    return cpu_time


# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data[0], data[1]
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        
        if(os.path.exists(f"client_properties_{self.cid}.pickle")):
            cpkl = open(f"client_properties_{self.cid}.pickle", 'rb')
            data = pkl.load(cpkl)
            self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray", "cpu_time": data['cpu_time']}
        else:
            self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # print("construction: self.properties", self.properties)

        # instantiate model
        # self.net = Net()

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, net=None):
        if(net == None):
            net = Net()
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self):
        return self.properties
    
    def set_parameters(self, parameters):
        net = Net()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net
        
    def fit(self, parameters, config):
        
        # print(f"fit() on client cid={self.cid}")
        net = self.set_parameters(parameters)
        
        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
        )
        
        # send model to device
        net.to(self.device)
        
        # train
        cpu_time = train(net, trainloader, epochs=int(config["epochs"]), device=self.device)
        self.properties['cpu_time'] = cpu_time
        print("properties:", self.properties)
        
        f = open(f"client_properties_{self.cid}.pickle",'wb')
        pkl.dump(self.properties, f)
        f.close()
        
        # return local model and statistics
        return self.get_parameters(net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(5),
        "batch_size": str(64),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 4. Starts a Ray-based simulation where a % of clients are sample each round.
# 5. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    pool_size = 100  # number of dataset partions (= number of total clients)
    client_resources = {"num_cpus": 1}  # each client will get allocated 1 CPUs

    # download CIFAR10 dataset
    train_path, testset = getCIFAR10()

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated: in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )

    # configure the strategy
    strategy = fl.server.strategy.Selective_FedAvg(
        fraction_fit=0.1,
        min_fit_clients=10,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),  # centralised testset evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return CifarRayClient(cid, fed_dir)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=5,
        strategy=strategy,
        ray_init_args=ray_config,
    )
