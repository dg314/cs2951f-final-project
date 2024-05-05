import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(PolicyNetwork, self).__init__()
        self.linear1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.linear2: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.softmax(self.linear2(x), dim=-1)

        return x
    
def train_policy_network(network: PolicyNetwork, observations: np.ndarray, actions: np.ndarray, learning_rate: int = 0.001, epochs: int = 10, batch_size: int = 32) -> None:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(observations, dtype=torch.float32), torch.tensor(actions, dtype=torch.long))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0

        for obs, action in data_loader:
            optimizer.zero_grad()

            output = network(obs)
            loss = loss_fn(output, action)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * obs.size(0)
        
        epoch_loss /= len(dataset)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

def policy_network_to_action(policy_network: PolicyNetwork, obs: np.ndarray) -> int:
    with torch.no_grad():
        output = policy_network(torch.tensor([obs], dtype=torch.float32))
        action = torch.argmax(output, dim=-1).item()

        return action