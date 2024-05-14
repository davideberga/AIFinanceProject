from __future__ import division
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shutil
import torch.autograd as Variable

from lib.IVVEnvironment import Actions

EPS = 0.003
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


device = "cpu" if not torch.cuda.is_available() else 'cuda'


class Critic(nn.Module):

    def __init__(self, feature_size, action_size, batch_size, is_eval=False):
        super(Critic, self).__init__()
        self.hidden_size = 256
        self.num_layers = 2
        self.is_eval = is_eval
        self.batch_size = batch_size
        self.relu = nn.ReLU()

        self.gruState = nn.GRU(feature_size,
                               self.hidden_size,
                               self.num_layers,
                               batch_first=True,
                               device=device,
                               dropout=0 if self.is_eval else 0.5)

        self.fc_state = nn.Linear(self.hidden_size, 128)
        self.fc_action = nn.Linear(action_size, 128)

        self.fc_action_state = nn.Linear(256, 128)
        self.fc_action_state.weight.data = fanin_init(
            self.fc_action_state.weight.data.size())

        self.fc_out = nn.Linear(128, 1)
        self.fc_out.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action, batch_size):

        batch_size_r = batch_size if batch_size != None else self.batch_size
        initial_hidden = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float32).to(
            device) if batch_size_r == 0 else torch.zeros(self.num_layers, state.shape[0], self.hidden_size, dtype=torch.float32).to(device)
        state, _ = self.gruState(state, initial_hidden)
        state = self.relu(state[-1] if batch_size_r == 0 else state[:, -1, :])
        state = self.relu(self.fc_state(state))

        action = self.relu(self.fc_action(action))
        action_state = torch.cat((state, action), dim=1)
        action_state = self.relu(self.fc_action_state(action_state))

        return self.fc_out(action_state)


class Actor(nn.Module):

    def __init__(self, feature_size, action_size, action_lim, batch_size, is_eval=False):
        super(Actor, self).__init__()
        self.hidden_size = 256
        self.num_layers = 2
        self.state_dim = feature_size
        self.action_dim = action_size
        self.action_lim = action_lim
        self.is_eval = is_eval
        self.relu = nn.ReLU()
        self.batch_size = batch_size

        self.gruState = nn.GRU(self.state_dim,
                               self.hidden_size,
                               self.num_layers,
                               batch_first=True,
                               device=device,
                               dropout=0 if self.is_eval else 0.5)

        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.action_out = nn.Linear(128, 1)
        # self.action_out.weight.data.uniform_(-EPS,EPS)

    def forward(self, state, batch_size=None):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """

        batch_size_r = batch_size if batch_size != None else self.batch_size
        initial_hidden = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float32).to(
            device) if batch_size_r == 0 else torch.zeros(self.num_layers, state.shape[0], self.hidden_size, dtype=torch.float32).to(device)
        state, _ = self.gruState(state, initial_hidden)
        state = state[-1] if batch_size_r == 0 else state[:, -1, :]

        state = F.tanh(self.fc1(state))
        action = F.tanh(self.action_out(state))

        action = action * self.action_lim

        return action


class A2CAgent:

    def __init__(self, state_dim, action_dim, action_lim, ram, batch_size):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.iter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)
        self.batch_size = batch_size

        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_lim, batch_size)
        self.actor.to(device)
        self.target_actor = Actor(
            self.state_dim, self.action_dim, self.action_lim, batch_size)
        self.target_actor.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 0.001)

        self.critic = Critic(self.state_dim, self.action_dim, batch_size)
        self.critic.to(device)
        self.target_critic = Critic(
            self.state_dim, self.action_dim, batch_size)
        self.target_critic.to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 0.001)
        self.agent_inventory = []

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.from_numpy(state).to(device)
        action = self.target_actor.forward(state).detach()
        cpu_action = action.cpu().numpy()

        action_type = lambda x: Actions.HOLD.value if x > -1 and x < 1 else Actions.SELL.value if x < -1 else Actions.BUY.value

        if action_type(cpu_action) != Actions.HOLD.value and len(self.agent_inventory) > 0:
            last_action = self.agent_inventory.pop(0)
            if last_action == action_type(cpu_action):  # if BUY BUY, or SELL SELL, the action is HOLD
                cpu_action *= -1
                # self.agent_inventory.append(last_action)

        elif action_type(cpu_action) != Actions.HOLD.value:
            self.agent_inventory.append(action_type(cpu_action))

        return cpu_action

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        return new_action

    def reset_inventory(self):
        self.agent_inventory = []

    def get_inventory(self):
        return self.agent_inventory

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2 = self.ram.sample(BATCH_SIZE)

        s1 = torch.from_numpy(s1).to(device)
        a1 = torch.from_numpy(a1).to(device)
        r1 = torch.from_numpy(r1).to(device)
        s2 = torch.from_numpy(s2).to(device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2, BATCH_SIZE).detach()
        next_val = torch.squeeze(
            self.target_critic.forward(s2, a2, BATCH_SIZE).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA*next_val
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1, BATCH_SIZE))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1, BATCH_SIZE)
        loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1, BATCH_SIZE))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

        # if self.iter % 100 == 0:
        # 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
        # 		' Loss_critic :- ', loss_critic.data.numpy()
        # self.iter += 1


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
    ou = OrnsteinUhlenbeckActionNoise(1)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()


class MemoryBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def len(self):
        return self.len

    def add(self, s, a, r, s1):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s, a, r, s1)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)
