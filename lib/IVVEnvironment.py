from enum import Enum
import gym
from gym import spaces
from gym.utils import seeding
from lib.IVVDataLoader import IVVDataset
from torch.utils.data import DataLoader
import torch

class Actions(Enum):
    HOLD = 0
    SELL = 1
    BUY = 2

class Positions(Enum):
    SHORT = 0
    LONG = 1

class IVVEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides window_size observations for a stock price series
    An episode is defined as a sequence of 390 minutes, from 1 minute timeframe data in day.
    It is possible that not every day contains so much data

    Each minute is a 'step' that allows the agent to choose one of three actions:
    - 0: HOLD
    - 1: BUY
    - 2: SELL
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data_path: str,
                 seed: int,
                 device: str,
                 trading_cost=1e-3,
                 action_size=3,
                 window_size=10):
        
        self.trading_cost = trading_cost
        self.window_size = window_size
        self.device = device

        self.dataset = IVVDataset(data_path, ['Volume', 'High'])
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.action_space = spaces.Discrete(action_size)
        self.seed(seed)

        self.day_number = 0
        self.episode_minute = 0

        self.total_profit = 0
        self.inventory = []
        self.when_sold = []
        self.when_bought = []

        self.buy_sell_order = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    '''
        Returns state observation, reward, done and info
    '''
    def step(self, action):
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        
        current_episode = self.dataset[self.day_number].squeeze()
        current_price = current_episode[self.episode_minute][2]
        episode_length = current_episode.shape[0]

        done = self.episode_minute == episode_length -1
        observation = self._get_observation(self.episode_minute)
        
        reward = 0
        if action == 1:
            self.inventory.append(current_price)
            self.when_bought.append(self.episode_minute)
            self.buy_sell_order.append('BUY')

        elif action == 2 and len(self.inventory) > 0:
            bought_price = self.inventory.pop(0)      
            reward = current_price - bought_price
            self.total_profit += current_price - bought_price
            self.when_sold.append(self.episode_minute)
            self.buy_sell_order.append('SELL')

        info = {
            'total_profit' : self.total_profit,
            'when_sold': self.when_sold,
            'when_bought': self.when_bought,
            'buy_sell_order': self.buy_sell_order,
        }

        self.episode_minute += 1

        return observation, reward, done, info
    
    def _get_observation(self, current_minute):    
        data = self.dataset[self.day_number].squeeze()
        data = torch.from_numpy(data).to(self.device)
        from_minute = current_minute - self.window_size + 1

        if from_minute>=0:
            block = data[from_minute:current_minute + 1, :]
        else:
            block = torch.cat([ data[0].repeat(-from_minute, 1), data[0:current_minute + 1, :] ])

        res = []
        for i in range(self.window_size - 1):
            res.append(block[i + 1].cpu().numpy() - block[i].cpu().numpy())

        return torch.transpose(torch.tensor(res).to(self.device), 0, 1)
    
    def there_is_another_episode(self):
        return self.day_number <= len(self.dataloader) - 1
    
    def num_of_ep(self):
        return len(self.dataloader)

    '''
        Jump to the next day/episode, 
    '''
    def reset(self):
        self.day_number += 1
        self.episode_minute = 0

        self.total_profit = 0
        self.inventory = []
        self.when_sold = []
        self.when_bought = []

        self.buy_sell_order = []

        return self._get_observation(self.episode_minute)
    
    '''
        Force the environment to start from day 0
    '''
    def close(self):
        self.day_number = 0
        self.episode_minute = 0

        self.total_profit = 0
        self.inventory = []
        self.when_sold = []
        self.when_bought = []

        self.buy_sell_order = []

    # TODO
    def render(self, mode='human'):
        """Not implemented"""
        pass