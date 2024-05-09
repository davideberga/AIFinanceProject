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
feature_size = 2
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

    def reset_invetory(self):
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
        batch = [[], [], [], [], []]
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
            batch[0].append(np.transpose(self.memory[i][0].cpu().numpy()))
            batch[1].append(torch.tensor(self.memory[i][1]))
            batch[2].append(torch.tensor(self.memory[i][2]))
            batch[3].append(self.memory[i][3])
            batch[4].append(torch.tensor(self.memory[i][4]))
        
        for _ in range(times_shuffle):
            random.shuffle(mini_batch)
            
            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s, batch[4])), device=device, dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in batch[3] if s is not None])
            
            state_batch = torch.from_numpy(np.array(batch[0]))
            action_batch =  torch.transpose(torch.from_numpy(np.array(batch[1])), 0, -1)
            reward_batch =  torch.from_numpy(np.array(batch[2]))

            print(state_batch.shape)
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)

            state_action_values =  self.model(state_batch.float()).gather(1, action_batch)

            next_state_values = torch.zeros(batch_size, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.model(non_final_next_states.float()).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            loss = nn.MSELoss()
            output = loss(expected_state_action_values, state_action_values)
            output.backward()
            self.optimizer.step()

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

            total_profit_loss += info['net_profit']

            if done: break

            observation = next_observation

        episode_count+=1

    print(f'Validation finished! Val profit, Model from ep {current_episode} on {episode_count} days: {total_profit_loss}')

agent = Agent(feature_size, window_size)
train_environment = IVVEnvironment(train_path, seed=seed, device=device, trading_cost=1e-3)

episode_count = 0
rewards_list = []
profit_loss_list = []

times_update_dqn = 1

curr_val_thread : threading.Thread = None
while(train_environment.there_is_another_episode()):
    # print("Running episode " + str(episode_count) + "/" + str(train_environment.num_of_ep()))

    # Reset the environment and obtain the initial observation
    observation = train_environment.reset()
    info = {}
    net_profit = []


    episode_loss = 0
    start_time = time.time()
    agent.reset_invetory()

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
        perform_validation(agent, episode_count, 285)
        # freezed_agent = copy.deepcopy(agent)
        # curr_val_thread = threading.Thread(target=perform_validation, args=(freezed_agent, episode_count, 285), daemon=True)
        # curr_val_thread.start()
        

    # plot_behavior(day_episode, states_buy, states_sell, total_profit)
    # print(f" >>> Episode: {episode_count} Reward: {np.mean(rewards_list):3.5f} Loss: {str(episode_loss)}, \n >>> Profit: {info['total_profit']}, BUY trades: {len(info['when_bought'])}, SELL trades: {len(info['when_sold'])}, \n >>> Time : {str(time.time() - start_time)}")
    # print(info['buy_sell_order'])

    #Calculate Success Rate metric
    successRate = info["positive_trades"]/len(info["when_sold"]) if len(info["when_sold"]) != 0  else 0
    print('Success Rate: ', successRate)

    #Calculate Net Return metric
    netReturn = sum(net_profit)/len(net_profit)
    print('Net Return: ', netReturn)

    #Calculate Sharpe Ratio metric
    net_profit = np.array(net_profit)
    sharpeRatio = (np.mean(net_profit) / np.std(net_profit))# - risk_free_rate
    print('Sharpe Ratio: ', sharpeRatio)

    #Calculate Maximum Drawdown
    rollingMax = torch.cummax(torch.tensor(net_profit), 0).values
    max_drawdown = torch.tensor(net_profit) - rollingMax
    max_drawdown_percentage = (max_drawdown / rollingMax).mean()
    print('Maximum Drawdown percentage: ', max_drawdown_percentage.item(), '%')
    
    print(f" >>> Episode: {episode_count} Reward: {np.mean(rewards_list):3.5f} BUY trades: {len(info['when_bought'])}, SELL trades: {len(info['when_sold'])}, Time: {time.time()-start_time}")
