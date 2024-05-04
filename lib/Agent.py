import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AgentNetwork(nn.Module, ABC):

    def __init__(self, feature_size, window_size, action_size):
        super(AgentNetwork, self).__init__()
        self.feature_size = feature_size
        self.window_size = window_size
        self.action_size = action_size

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()
    
    # def save_model(self, path: str):
    #     torch.save(self.state_dict(), path)

class AgentCNNNetwork(AgentNetwork):

    def __init__(self, feature_size, window_size, action_size):
        super(AgentCNNNetwork, self).__init__(feature_size, window_size, action_size)
        self.conv1 = nn.Conv1d(in_channels=self.feature_size, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4, self.action_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return self.fc1(x)