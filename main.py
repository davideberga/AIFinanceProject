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
from gym.wrappers import FlattenObservation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from collections import deque
import torch
from empyrical import max_drawdown, cum_returns_final, sharpe_ratio, cagr, annual_volatility, value_at_risk, conditional_value_at_risk

device = "cpu" if not torch.cuda.is_available() else 'cuda'
#Disable the warnings
import warnings
warnings.filterwarnings('ignore')

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

window_size = 64
batch_size = 64
feature_size = 4
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

        self.model = AgentLSTMNetwork(self.feature_size, self.window_size, self.action_size, device, is_eval)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.agent_inventory = []

    def reset_invetory(self):
        self.agent_inventory = []

    def evaluation_mode(self, eval=True):
        self.is_eval= eval

    def act(self, state): 
        #If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
        #actions suggested.

        action_selected = 0
        self.model.eval()
        action_by_model = 0
        # Epsilon-greedy policy only on training
        if not self.is_eval and random.random() <= self.epsilon:
            action_selected = random.randrange(self.action_size) 
        else:
            with torch.no_grad():
                state = torch.transpose(state, 0, 1).to(device)
                options = self.model(state.float(), 0).reshape(-1).cpu().numpy()   
                action_selected = np.argmax(options)
                action_by_model = 1 if action_selected != 0 else 0

        # Force correct action only in evaluation mode
        if self.is_eval:
            if action_selected != 0 and len(self.agent_inventory) > 0:
                last_action = self.agent_inventory.pop(0)
                if last_action == action_selected: # if BUY BUY, or SELL SELL, the action is HOLD
                    action_selected = 0
                    self.agent_inventory.append(last_action)
            elif action_selected != 0:
                self.agent_inventory.append(action_selected)
        
        return action_selected, action_by_model

    def expReplay(self, batch_size, times_shuffle=1):

        self.model.train()

        exp_repl_mean_loss = 0
        for _ in range(times_shuffle):

            batch = [[], [], [], [], []]
            sample = random.sample(self.memory, batch_size)
            for observation, action, reward, next_observation, done in sample:
                batch[0].append(np.transpose(observation.cpu().numpy()))
                batch[1].append(torch.tensor(action))
                batch[2].append(torch.tensor(reward))
                batch[3].append(np.transpose(next_observation.cpu().numpy()))
                batch[4].append(torch.tensor(done))
            
            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch[4])), device=device, dtype=torch.bool)
            non_final_next_states = torch.from_numpy(np.array(batch[3])).to(device)[non_final_mask]

            # non_final_next_states = torch.cat([s for s in next_states_batch if s is not None])
            
            state_batch = torch.from_numpy(np.array(batch[0]))
            action_batch =  torch.from_numpy(np.array(batch[1])).view(-1, 1)
            reward_batch =  torch.from_numpy(np.array(batch[2]))

            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)
            non_final_next_states = non_final_next_states.to(device)


            state_action_values =  self.model(state_batch.float(), state_batch.shape[0]).gather(1, action_batch)

            next_state_values = torch.zeros(batch_size, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.model(non_final_next_states.float(), non_final_next_states.shape[0]).max(1).values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            loss = nn.MSELoss()
            output = loss(expected_state_action_values.float(), state_action_values.float())
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

    total_net_profit = []
    total_profit_loss = []
    episode_count = 0
    actions_by_model = 0
    
    print(f'Start validation: model from ep {current_episode} on {max_episodes} days')
    while(validation_environment.there_is_another_episode() and (max_episodes == -1 or episode_count < max_episodes)):

        # Reset the environment and obtain the initial observation
        observation = validation_environment.reset()
        agent.reset_invetory()
        info = {}

        while True:

            action, action_by_model = agent.act(observation)
            actions_by_model += action_by_model
            next_observation, reward, done, info = validation_environment.step(action)
            
            if info['net_profit'] != 0:
                total_net_profit.append(info['net_profit'])
            if info['profit_loss'] != 0:
                total_profit_loss.append(info['profit_loss'])

            if done: break

            observation = next_observation

        episode_count+=1
        print(episode_count)
        #Ogni 285 episodi/giorni calcolo le metriche: quindi calcolo le metriche ogni anno
        if episode_count % 285 == 0:

            net_profit = np.array(total_net_profit)
            profit_loss = np.array(total_profit_loss)

            print("è passato un anno")
            #Calculate Sharpe Ratio metric
            sharpeRatio = sharpe_ratio(profit_loss,risk_free=0)
            print('Sharpe Ratio: ', sharpeRatio)

            #Calculate Maximum Drawdown metric
            maxDrawdown = max_drawdown(profit_loss)
            print('Maximum Drawdown: ', maxDrawdown)

            #Calculate Compounded Annual Return metric
            annualReturn = cagr(profit_loss, annualization=1)
            print('Compounded Annual Return: ', annualReturn)

            #Calculate Annual Volatility metric
            annualVolatility = annual_volatility(profit_loss, annualization=1)
            print('Annual Volatility: ', annualVolatility)

            #Calculate Value at Risk metric
            valueAtRisk = value_at_risk(profit_loss)
            print('Value at Risk: ', valueAtRisk)

            #Calculate Conditional Value at Risk metric
            condValueAtRisk = conditional_value_at_risk(profit_loss)
            print('Conditional Value at Risk: ', condValueAtRisk)

    print(f'Actions by model {str(actions_by_model)}')
    print(f'>>> Validation finished! <<< \n')

# -------- Validation finished -----------

# -------- Train -----------
agent = Agent(feature_size, window_size)
train_environment = IVVEnvironment(train_path, seed=seed, device=device, window_size=window_size, trading_cost=1e-3)

episode_count = 0
rewards_list = []
times_update_dqn = 3

while(train_environment.there_is_another_episode()):
    # print("Running episode " + str(episode_count) + "/" + str(train_environment.num_of_ep()))

    # Reset the environment and obtain the initial observation
    observation = train_environment.reset()
    info = {}
    net_profit = []
    profit_loss = []


    episode_loss = 0
    start_time = time.time()
    agent.reset_invetory()
    actions_by_model = 0

    while True:

        action, action_by_model = agent.act(observation)
        actions_by_model += action_by_model
        next_observation, reward, done, info = train_environment.step(action)
       
        if info["net_profit"] != 0:
            net_profit.append(info["net_profit"])
        if info["profit_loss"] != 0:
            profit_loss.append(info["profit_loss"])

        agent.memory.append((observation, action, reward, next_observation, done))
        rewards_list.append(reward)

        if done: break

        if len(agent.memory) > batch_size:
            episode_loss += agent.expReplay(batch_size, times_update_dqn) 

        observation = next_observation

    episode_count += 1

    if episode_count % 20 == 0:
        agent.evaluation_mode()
        perform_validation(agent, episode_count, 120)
        agent.evaluation_mode(False)
        # freezed_agent = copy.deepcopy(agent)
        # curr_val_thread = threading.Thread(target=perform_validation, args=(freezed_agent, episode_count, 285), daemon=True)
        # curr_val_thread.start()
        

    # plot_behavior(day_episode, states_buy, states_sell, total_profit)
    # print(f" >>> Episode: {episode_count} Reward: {np.mean(rewards_list):3.5f} Loss: {str(episode_loss)}, \n >>> Profit: {info['total_profit']}, BUY trades: {len(info['when_bought'])}, SELL trades: {len(info['when_sold'])}, \n >>> Time : {str(time.time() - start_time)}")
    # print(info['buy_sell_order'])

    net_profit = np.array(net_profit)
    profit_loss = np.array(profit_loss)

    print(f">>> EPISODE: {episode_count} <<<")
    print(f"Reward: {np.mean(rewards_list):3.5f}, Profits: {sum(profit_loss)}, BUY: {len(info['when_bought'])}, SELL: {len(info['when_sold'])}") #, Time: {time.time()-start_time}")
    
    print("---- Metrics ----")

    #Calculate Success Rate metric
    print(f'Actions by model {str(actions_by_model)}')
    if len(info["when_sold"]) > len(info["when_bought"]):
        successRate = info["positive_trades"]/len(info["when_sold"]) if len(info["when_sold"]) != 0  else 0
        print('Success Rate: ', successRate)
    else:
        successRate = info["positive_trades"]/len(info["when_bought"]) if len(info["when_bought"]) != 0  else 0
        print('Success Rate: ', successRate)
    

    #Calculate Net Return metric
    if len(net_profit) == 0:
        netReturn = 0
    else:
        netReturn = sum(net_profit)/len(net_profit)
    print('Net Return: ', netReturn)

    #Calculate Sharpe Ratio metric
    #sharpeRatio1 = (np.mean(net_profit) / np.std(net_profit))# - risk_free_rate
    #print('Sharpe Ratio1: ', sharpeRatio1)

    #sharpeRatio2 = sharpe_ratio(net_profit,risk_free=0)
    #print('Sharpe Ratio2: ', sharpeRatio2)

    #Calculate Maximum Drawdown metric
    #maxDrawdown = max_drawdown(profit_loss_list)
    #print('Maximum Drawdown: ', maxDrawdown)

    #Calculate Cumulative Returns metric:
    #cumReturns = cum_returns_final(profit_loss_list)
    #print('Cumulative Returns: ', cumReturns)
    
    print("\n")

perform_validation(agent, episode_count, 750)