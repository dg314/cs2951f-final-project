import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable

class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(PolicyNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.linear2: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor):
        output = torch.relu(self.linear1(input))
        output = torch.softmax(self.linear2(output), dim=-1)

        return output
    
def train_policy_network(policy_network: PolicyNetwork, observations: np.ndarray, actions: np.ndarray, learning_rate: float = 0.001, epochs: int = 10, batch_size: int = 32) -> None:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(observations, dtype=torch.float32), torch.tensor(actions, dtype=torch.long))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0

        for input, label in data_loader:
            optimizer.zero_grad()

            output = policy_network(input)
            loss = loss_fn(output, label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * input.size(0)
        
        epoch_loss /= len(dataset)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

def policy_network_to_action(policy_network: PolicyNetwork, obs: np.ndarray) -> int:
    with torch.no_grad():
        output = policy_network(torch.tensor(np.array([obs]), dtype=torch.float32))
        action = torch.argmax(output, dim=-1).item()

        return action
    
def policy_networks_to_action(policy_networks: list[PolicyNetwork], ratings: list[float], obs: np.ndarray) -> int:
    with torch.no_grad():
        rated_outputs = [
            policy_network(torch.tensor(np.array([obs]), dtype=torch.float32)) * rating
            for policy_network, rating in zip(policy_networks, ratings)
        ]
        rated_output_sum = torch.sum(torch.stack(rated_outputs, dim=0), dim=0)
        action = torch.argmax(rated_output_sum, dim=-1)

        return action
    
class Discriminator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(Discriminator, self).__init__()
        self.linear1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.linear2: nn.Linear = nn.Linear(hidden_size, 1)

    def forward(self, input: torch.Tensor):
        output = torch.relu(self.linear1(input))
        output = torch.sigmoid(self.linear2(output))

        return output
    
def train_gan(generator: PolicyNetwork, discriminator: Discriminator, sample_demo_trajectories: Callable[[], np.ndarray], sample_generator_observations: Callable[[], np.ndarray], learning_rate: float = 0.001, epochs: int = 10, batch_size: int = 1024) -> None:
    discriminator_loss_fn = nn.BCELoss()
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        discriminator_epoch_loss = 0
        generator_epoch_loss = 0

        demo_trajectories = torch.tensor(sample_demo_trajectories(), dtype=torch.float32)
        generator_observations = torch.tensor(sample_generator_observations(), dtype=torch.float32)
        
        generator_actions = generator(generator_observations)
        generator_trajectories = torch.cat((generator_observations, generator_actions), dim=1)

        discriminator_inputs = torch.cat((demo_trajectories, generator_trajectories.detach()), dim=0)
        discriminator_labels = torch.cat((torch.ones(demo_trajectories.shape[0], 1), torch.zeros(generator_trajectories.shape[0], 1)), dim=0)

        dataset = TensorDataset(discriminator_inputs, discriminator_labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for input, label in data_loader:
            discriminator_optimizer.zero_grad()

            output = discriminator(input)
            discriminator_loss = discriminator_loss_fn(output, label)

            discriminator_loss.backward()
            discriminator_optimizer.step()

            discriminator_epoch_loss += discriminator_loss.item() * input.size(0)

        generator_dataset = TensorDataset(generator_observations)
        generator_data_loader = DataLoader(generator_dataset, batch_size=batch_size, shuffle=True)

        for obs, in generator_data_loader:
            generator_optimizer.zero_grad()

            action = generator(obs)
            trajectory = torch.cat((obs, action), dim=1)
            discriminator_output = discriminator(trajectory)

            generator_loss = -torch.mean(torch.log(discriminator_output + 1e-10))
            generator_loss.backward()
            generator_optimizer.step()

            generator_epoch_loss += generator_loss.item() * obs.size(0)

        discriminator_epoch_loss /= len(discriminator_inputs)
        generator_epoch_loss /= len(generator_observations)

        print(f"Epoch [{epoch+1}/{epochs}], Discriminator Loss: {discriminator_epoch_loss:.4f}, Generator Loss: {generator_epoch_loss:.4f}")