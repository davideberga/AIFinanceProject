from enum import Enum
import gym
from gym import spaces
from gym.utils import seeding
from lib.IVVDataLoader import IVVDataset
from torch.utils.data import DataLoader
import torch

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
        self.positive_trades = 0
        self.profit_or_loss = []

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
        current_price = current_episode[self.episode_minute][2]
        episode_length = current_episode.shape[0]

        done = self.episode_minute == episode_length -1
        observation = self._get_observation(self.episode_minute)
        if done and len(self.inventory) > 0: action = 1 if self.inventory[0][0] == 2 else 2
        
        reward = 0
        if action == Actions.BUY.value:
            self.when_bought.append(self.episode_minute)
            self.buy_sell_order.append('BUY')
        elif (action == Actions.SELL.value or done):
            self.when_sold.append(self.episode_minute)
            self.buy_sell_order.append('SELL') 

        reward, profit_loss = self.update_reward(current_price, action)
        self.positive_trades += int(profit_loss > 0)
        self.total_profit += profit_loss
            
        #Calculate transactional costs for each trade
        transaction_costs = self._calculate_transaction_costs(current_price, 0.001, 0.01)

        #Calculate Net Return metric
        net_profit = profit_loss - transaction_costs

        info = {
            'total_profit' : self.total_profit,
            'when_sold': self.when_sold,
            'when_bought': self.when_bought,
            'buy_sell_order': self.buy_sell_order,
            "positive_trades": self.positive_trades,
            "net_profit": net_profit
        }

        self.episode_minute += 1

        return observation, reward, done, info
    
    def update_reward(self, price, action):
        reward = 0
        profit_loss = 0

        if action == Actions.BUY.value:  # Buy
            if len(self.inventory) == 0:  # non ho posizioni aperte 
                self.inventory.append((Actions.BUY.value, price)) # aggiungo una posizione, con 1 l'azione buy e price il prezzo a cui ho comprato
            else: # ho già una posizione aperta, controllo qual è questa posizione
                action_done, position_open_price = self.inventory.pop(0)
                if action_done == Actions.SELL.value: # se posizione aperta di sell, va bene
                    reward, profit_loss = self.compute_reward(price, position_open_price, action)
                elif action_done == Actions.BUY.value: # se posizione aperta di buy, non va bene
                    reward = -100 # perdo totalmente i soldi di quell'azione
        
        elif action == Actions.SELL.value:  # Sell
            if len(self.inventory) == 0:  # non ho posizioni aperte
                self.inventory.append(( Actions.SELL.value, price))
            else: # ho già una posizione aperta
                action_done, position_open_price = self.inventory.pop(0)
                if action_done ==  Actions.BUY.value: # se posizione aperta di buy, va bene
                    reward, profit_loss = self.compute_reward(price, position_open_price, action)
                elif action_done ==  Actions.SELL.value: # se posizione aperta di sell, non va bene
                    reward = -100 # perdo totalmente i soldi di quell'azione

        return reward, profit_loss

    def compute_reward(self, price, open_price, action):
        if action == 1:  # Buy
            profit_loss = open_price - price  # open_price è il prezzo a cui ho venduto prima
        elif action == 2:  # Sell
            profit_loss = price - open_price  # open_price è il prezzo a cui ho comprato prima
        
        profit_loss = profit_loss - 0.01 * price # 1% del prezzo a quell'istante, quando si chiude la posizione
        risk = abs(profit_loss)
        
        reward = profit_loss - 0.5 * risk  # penalizziamo il rischio del 50%
        
        return 100 * reward, profit_loss
    
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
        self.capital = self.inital_capital

        self.positive_trades = 0
        self.profit_or_loss = []

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

        self.positive_trades = 0
        self.profit_or_loss = []
        self.capital = self.inital_capital

        self.buy_sell_order = []

    # TODO
    def render(self, mode='human'):
        """Not implemented"""
        pass