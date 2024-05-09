import copy
from lib.IVVEnvironment import IVVEnvironment
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy.random import choice
import random
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from collections import deque
import torch

device = "cpu" if not torch.cuda.is_available() else 'cuda'
#Disable the warnings
import warnings
warnings.filterwarnings('ignore')

import torch

from lib.AgentNetworks import AgentCNNNetwork, AgentLSTMNetwork, AgentGRUNetwork
import time

train_path = "lib/data/IVV_1m_training.csv"
validation_path = "lib/data/IVV_1m_validation.csv"

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# -------- Validation -----------
from lib.IVVEnvironment import IVVEnvironment

window_size = 10
batch_size = 16
feature_size = 3
seed = 9
seed_everything(9)

validation_environment = IVVEnvironment(validation_path, seed=seed, device=device, trading_cost=1e-3)

class Agent():
    def __init__(self, feature_size, window_size, is_eval=False, model_name=""):
        super(Agent, self).__init__()
        self.feature_size = feature_size
        self.window_size = window_size
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = AgentGRUNetwork(self.feature_size, self.window_size, self.action_size, device, is_eval)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.agent_inventory = []

    def act(self, state): 
        #If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
        #actions suggested.

        action_selected = 0
        self.model.eval()
        if not self.is_eval and random.random() <= self.epsilon:
            action_selected = random.randrange(self.action_size) 
        else:
            with torch.no_grad():
                options = self.model(state.float()).reshape(-1).cpu().numpy()   
                action_selected = np.argmax(options)

        if action_selected != 0 and len(self.agent_inventory) > 0:
            last_action = self.agent_inventory.pop(0)
            if last_action == action_selected: # if BUY BUY, or SELL SELL, the action is HOLD
                action_selected = 0
                self.agent_inventory.append(last_action)
        elif action_selected != 0:
            self.agent_inventory.append(action_selected)
        
        return action_selected

    def expReplay(self, batch_size, times_shuffle=1):
        mini_batch = []
        l = len(self.memory)

        self.model.train()
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        for _ in range(times_shuffle):
            random.shuffle(mini_batch)
            exp_repl_mean_loss = 0
            for state, action, reward, next_state, done in mini_batch:
                
                self.optimizer.zero_grad()
                output = self.model(state.float()).reshape(-1)
                # Target does not need gradients
                target_f = output.detach().clone()

                if done: 
                    target_f[action] = reward
                else:
                    with torch.no_grad():
                        target_f[action] = reward + self.gamma * torch.max(self.model(next_state.float()).reshape(-1)).cpu().numpy()   

                loss = nn.MSELoss()
                output = loss(target_f, output)
                output.backward()
                self.optimizer.step()

                exp_repl_mean_loss += output.item()

        exp_repl_mean_loss /= batch_size * times_shuffle
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return exp_repl_mean_loss
    
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

window_size = 10
batch_size = 16
feature_size = 3
seed = 9
seed_everything(9)

def perform_validation(agent: Agent, current_episode, max_episodes=-1):
    validation_environment.close()

    total_profit_loss = 0
    episode_count = 0
    print(f'Start validation: model from ep {current_episode} on {max_episodes} days')
    while(validation_environment.there_is_another_episode() and (max_episodes == -1 or episode_count < max_episodes)):

        # Reset the environment and obtain the initial observation
        observation = validation_environment.reset()
        info = {}

        while True:

            action = agent.act(observation)
            next_observation, reward, done, info = validation_environment.step(action)

            total_profit_loss += info['net_return']

            if done: break

            observation = next_observation

        episode_count+=1

    print(f'Validation finished! Val profit, Model from ep {current_episode} on {episode_count} days: {total_profit_loss}')

agent = Agent(feature_size, window_size)
train_environment = IVVEnvironment(train_path, seed=seed, device=device, trading_cost=1e-3)

episode_count = 0
rewards_list = []

times_update_dqn = 3

curr_val_thread : threading.Thread = None
while(train_environment.there_is_another_episode()):
    # print("Running episode " + str(episode_count) + "/" + str(train_environment.num_of_ep()))

    # Reset the environment and obtain the initial observation
    observation = train_environment.reset()
    info = {}
    net_profit = []

    episode_loss = 0
    start_time = time.time()
    while True:

        action = agent.act(observation)
        next_observation, reward, done, info = train_environment.step(action)
        net_profit.append(info["net_profit"])

        agent.memory.append((observation, action, reward, next_observation, done))
        rewards_list.append(reward)

        if done: break

        if len(agent.memory) > batch_size:
            episode_loss += agent.expReplay(batch_size, times_update_dqn) 

        observation = next_observation

    episode_count += 1

    if episode_count % 20 == 0:
        # freezed_agent = copy.deepcopy(agent)
        # curr_val_thread = threading.Thread(target=perform_validation, args=(freezed_agent, episode_count, 285), daemon=True)
        # curr_val_thread.start()
        perform_validation(agent, episode_count, 285)

    # plot_behavior(day_episode, states_buy, states_sell, total_profit)
    # print(f" >>> Episode: {episode_count} Reward: {np.mean(rewards_list):3.5f} Loss: {str(episode_loss)}, \n >>> Profit: {info['total_profit']}, BUY trades: {len(info['when_bought'])}, SELL trades: {len(info['when_sold'])}, \n >>> Time : {str(time.time() - start_time)}")
    # print(info['buy_sell_order'])

    #Calculate Success Rate metric
    successRate = info["positive_trades"]/len(info["when_sold"])
    print('Success Rate: ', successRate)

    #Calculate Net Return metric
    netReturn = sum(net_profit)/len(net_profit)
    print('Net Return: ', netReturn)

    #Calculate Sharpe Ratio metric
    rewards_list = np.array(rewards_list)
    sharpeRatio = (np.mean(rewards_list) / np.std(rewards_list))# - risk_free_rate
    print('Sharpe Ratio: ', sharpeRatio)

    #Calculate Maximum Drawdown
    rollingMax = torch.cummax(torch.tensor(net_profit), 0).values
    max_drawdown = torch.tensor(net_profit) - rollingMax
    max_drawdown_percentage = (max_drawdown / rollingMax).mean()
    print('Maximum Drawdown percentage: ', max_drawdown_percentage.item(), '%')
    
    episode_count += 1
    print(f" >>> Episode: {episode_count} Reward: {np.mean(rewards_list):3.5f} BUY trades: {len(info['when_bought'])}, SELL trades: {len(info['when_sold'])}, Time: {time.time()-start_time}")