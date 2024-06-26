import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.autograd import Variable

class AgentNetwork(nn.Module, ABC):

    '''
        feature_size: number of features 
        window_size: window length of past features
        action_size: number of actions to predict
    '''
    def __init__(self, feature_size, window_size, action_size, device, is_eval=False):
        super(AgentNetwork, self).__init__()
        self.feature_size = feature_size
        self.window_size = window_size
        self.action_size = action_size
        self.device = device
        self.to(device)
        self.is_eval = is_eval

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()
    
    # def save_model(self, path: str):
    #     torch.save(self.state_dict(), path)

class AgentCNNNetwork(AgentNetwork):

    def __init__(self, feature_size, window_size, action_size, device):
        super(AgentCNNNetwork, self).__init__(feature_size, window_size, action_size, device)
        self.conv1 = nn.Conv1d(in_channels=self.feature_size, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(26, self.action_size)

    def forward(self,x):
        x = torch.transpose(x, 1, 2).to(self.device)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.fc1(x)
        return x.squeeze(1)
    
class AgentLSTMNetwork(nn.Module):
    def __init__(self, feature_size, window_size, action_size, device):
        super(AgentLSTMNetwork, self).__init__()
        self.feature_size = feature_size
        self.window_size = window_size
        self.action_size = action_size
        self.device = device
        self.hidden_size = 128
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=self.feature_size, 
                            hidden_size=self.hidden_size,
                            batch_first=True)
        self.fc_1 = nn.Linear(self.hidden_size, 64)
        self.fc = nn.Linear(64, self.action_size)
        self.relu = nn.ReLU()

        self.to(device)

    def forward(self, x):
        output, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.relu(output[:, -1, :])
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        #print(out.shape)
        return out
    

class AgentGRUNetwork(AgentNetwork):

    def __init__(self, feature_size, window_size, action_size, device, is_eval=False):
        super(AgentGRUNetwork, self).__init__(feature_size, window_size, action_size, device, is_eval)
        self.hidden_size = 512
        self.num_layers = 1

        self.gru = nn.GRU(self.feature_size, 
                          self.hidden_size, 
                          self.num_layers, 
                          batch_first=True,
                          device=device)
        self.fc_1 = nn.Linear(512, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, self.action_size)
        self.relu = nn.ReLU()

        
    
    def forward(self,x, batch_size):
        initial_hidden = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float32).to(self.device) if batch_size == 0 else 0
        x, h = self.gru(x, initial_hidden)
        x = self.relu(self.fc_1( x[-1] if batch_size == 0 else x[:, -1, :]))
        x = self.relu(self.fc_2(x))
        return self.fc_out(x)