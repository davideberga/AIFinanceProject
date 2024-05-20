from enum import Enum
import gym
from gym import spaces
from gym.utils import seeding
from lib.IVVDataLoader import IVVDataset
from torch.utils.data import DataLoader
import torch
import numpy as np

class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

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

        self.dataset = IVVDataset(data_path, ['High', 'Low', 'Open', 'Volume'])
        self.action_space = spaces.Discrete(action_size)
        #self.seed(seed)
        self.moving_data = []

        self.day_number = -1
        self.episode_minute = 0

        self.total_profit = 0
        self.inventory = []
        self.when_sold = []
        self.when_bought = []
        self.positive_trades = 0
        self.profit_or_loss = []
        self.actual_trades = 0

        self.inital_capital = 10000
        self.capital = self.inital_capital

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
        current_price = current_episode[self.episode_minute][0]
        episode_length = current_episode.shape[0]

        done = self.episode_minute == episode_length - 1
        observation = self._get_observation(self.episode_minute)
        # if done and len(self.inventory) > 0: action = 1 if self.inventory[0][0] == 2 else 2
        
        
        reward = 0
        if action == Actions.BUY.value:
            self.when_bought.append(self.episode_minute)
            self.buy_sell_order.append('BUY')
        elif action == Actions.SELL.value:
            self.when_sold.append(self.episode_minute)
            self.buy_sell_order.append('SELL') 

        reward, profit_loss, net_profit = self.update_reward(current_price, action)

        self.positive_trades += int(profit_loss > 0)
        self.total_profit += profit_loss

        total_trades = len(self.when_bought) + len(self.when_sold)
        if done and total_trades <= 8 and total_trades >= 2: reward += 0
        if done and total_trades > 8: reward += - (total_trades - 8) * 10
        if done and total_trades < 2: reward += - 200

        
        # if total_trades % 2 == 0: reward += 5


        info = {
            'total_profit' : self.total_profit,
            'when_sold': self.when_sold,
            'when_bought': self.when_bought,
            'buy_sell_order': self.buy_sell_order,
            "positive_trades": self.positive_trades,
            "net_profit": net_profit,
            "profit_loss": profit_loss,
            "actual_trades": self.actual_trades
        }
        self.episode_minute += 1
        return observation, reward, done, info


    def update_reward(self, price, action):
        scaler = 100
        reward = 0
        profit_loss = 0
        net_profit = 0

        previous_price = self.dataset[self.day_number][self.episode_minute - (1 if self.episode_minute != 0 else 0)][0]
        # if action == Actions.SELL.value:
        #     reward = previous_price - price
        # elif action == Actions.BUY.value:
        #     reward = price - previous_price
        # else:
        #     reward = 0.01

        if len(self.inventory) == 0:
            if action == Actions.BUY.value:
                self.inventory.append((Actions.BUY.value, price))
                self.actual_trades += 1
            elif action == Actions.SELL.value:
                self.inventory.append((Actions.SELL.value, price))
                self.actual_trades += 1
        else:
            previous_action, open_position_price = self.inventory[0]
            if action == Actions.HOLD.value: reward = 0
            else:
                if action != previous_action:
                    self.inventory.pop(0)
                    self.actual_trades += 1
                    if action == Actions.SELL.value:
                        profit_loss = price - open_position_price
                    elif action == Actions.BUY.value:
                        profit_loss = open_position_price - price

                    transaction_costs = self._calculate_transaction_costs(price, self.trading_cost, 0.01) + self._calculate_transaction_costs(open_position_price, self.trading_cost, 0.01)
                    net_profit = profit_loss - transaction_costs
                    if profit_loss > 0: reward = profit_loss
                    self.total_profit += net_profit
                    self.profit_or_loss.append(net_profit)

        # if len(self.inventory) == 0:
        #     if action == Actions.BUY.value:
        #         self.inventory.append((Actions.BUY.value, price))
        #         self.actual_trades += 1
        #         reward += - self._calculate_transaction_costs(price, self.trading_cost, 0.01)
        #     elif action == Actions.SELL.value:
        #         self.inventory.append((Actions.SELL.value, price))
        #         self.actual_trades += 1
        #         reward += - self._calculate_transaction_costs(price, self.trading_cost, 0.01)
        # else:
        #     previous_action, open_position_price = self.inventory[0]
        #     if action == Actions.HOLD.value:
        #         if previous_action == Actions.BUY.value:
        #             # If price increase after BUY action, HOLD is a smart thing to do
        #             reward = scaler * (price - open_position_price)
        #         elif previous_action == Actions.SELL.value:
        #             # If price decrese after SELL action, so HOLD is a smart thing to do
        #             reward = scaler * (open_position_price - price)
        #     else:
        #         if action != previous_action:
        #             self.inventory.pop(0)
        #             self.actual_trades += 1
        #             if action == Actions.SELL.value:
        #                 profit_loss = price - open_position_price
        #             elif action == Actions.BUY.value:
        #                 profit_loss = open_position_price - price

        #             transaction_costs = self._calculate_transaction_costs(price, self.trading_cost, 0.01) + self._calculate_transaction_costs(open_position_price, self.trading_cost, 0.01)
        #             net_profit = profit_loss - transaction_costs
        #             reward = scaler * profit_loss * 20
        #             self.total_profit += net_profit
        #             self.profit_or_loss.append(net_profit)
        #         else:
        #             reward += - self._calculate_transaction_costs(price, self.trading_cost, 0.01) * scaler
                    
        return reward, profit_loss, net_profit


    
    def _get_observation(self, current_minute):    
        data = self.dataset[self.day_number]


        feature_size = 4

        last_open_action = 0 if len(self.inventory) == 0 else ( 1 if self.inventory[0][0] == Actions.BUY.value else -1)
        last_open_price =  0 if len(self.inventory) == 0 else self.inventory[0][1]
        if self.day_number == 0 and self.window_size - current_minute > 0:
            for _ in range(abs(self.window_size - current_minute)):
                self.moving_data.append([0 for _ in range(feature_size)])

        is_increasing = data[current_minute][0]-data[current_minute - self.window_size if current_minute > 0 else 0][0]
        is_increasing = is_increasing > 0

        self.moving_data.append([data[current_minute][0]-data[current_minute - 1 if current_minute > 0 else 0][0],
                                 is_increasing,
                                 last_open_action,
                                 len(self.when_sold) + len(self.when_bought)
                        #    len(self.when_sold) + len(self.when_bought)
                        #   # last_open_price, # ???? 
                        #   # is_increasing, # Trend
                        #   # risk
                        #     len(self.when_sold),
                        #     len(self.when_bought)
                        #   (len(self.when_sold) + len(self.when_bought)) % 2,
                          ])
        
        current_window = torch.transpose(torch.tensor(self.moving_data[-self.window_size:]).to(self.device), 0, 1)

        # mask = torch.tensor([[j==0 for _ in range(self.window_size)] for j in range(feature_size)]).to(self.device)
        # masked_window_mean = current_window[mask].clone().to(self.device)
        # masked_window_std = current_window[mask].clone().to(self.device)

        # mean = masked_window_mean.mean()
        # masked_window_mean -= mean
        # std = masked_window_std.std()
        # masked_window_mean /= std
        # current_window[mask] = masked_window_mean

        # print(current_window)

        return current_window
    
    def _calculate_transaction_costs(self, price, commission_rate, spread):
        # price: price of the "buy" or "sell"
        # commission_rate: scalar representing the commission rate per trade
        # spread: scalar representing the bid-ask spread --> The difference between the bid and ask prices of a security. 
        #                                                    When traders buy at the ask price and sell at the bid price, 
        #                                                    they incur a cost equal to the spread.

        # Calculate commission costs
        commission_costs = price * commission_rate

        # Calculate spread costs
        spread_costs = price * spread

        # Total transactional costs
        total_costs = commission_costs + spread_costs

        return total_costs
    
    def there_is_another_episode(self):
        return self.day_number <= len(self.dataset) - 2
    
    def num_of_ep(self):
        return len(self.dataset)

    '''
        Jump to the next day/episode, 
    '''
    def reset(self):
        self.day_number += 1
        self.episode_minute = 1
        self.actual_trades = 0
        self.total_profit = 0
        self.inventory = []
        self.when_sold = []
        self.when_bought = []
        self.positive_trades = 0
        self.profit_or_loss = []
        self.buy_sell_order = []
        return self._get_observation(0)
    
    '''
        Force the environment to start from day 0
    '''
    def close(self):
        self.day_number = -1
        self.episode_minute = 0
        self.moving_data = []

        self.total_profit = 0
        self.inventory = []
        self.when_sold = []
        self.when_bought = []

        self.positive_trades = 0
        self.profit_or_loss = []
        self.capital = self.inital_capital

        self.buy_sell_order = []

    def render(self, mode='human'):
        """Not implemented"""
        pass