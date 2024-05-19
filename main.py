import copy
from lib.DQNAgent import DQNAgent
from lib.IVVEnvironment import IVVEnvironment
# Load libraries
import numpy as np
import pandas as pd
from numpy.random import choice
import random
from gym.wrappers import FlattenObservation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
import torch
from empyrical import max_drawdown, sharpe_ratio, cagr, annual_volatility, value_at_risk, conditional_value_at_risk

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


window_size = 32
batch_size = 64
feature_size = 3
seed = 9
times_update_dqn = 3
TAU = 0.005

seed_everything(9)

train_environment = IVVEnvironment(train_path, seed=seed, device=device, window_size=window_size, trading_cost=1e-3)
validation_environment = IVVEnvironment(train_path, seed=seed, device=device, window_size=window_size, trading_cost=1e-3)
agent = DQNAgent(feature_size, window_size)
    
def perform_validation(current_episode, max_episodes=-1):
    validation_environment.close()

    total_net_profit = []
    total_profit_loss = []
    episode_count = 0
    actions_by_model = 0
    buy_actions = 0
    sell_actions = 0
    
    print(f'Start validation: model from ep {current_episode} on {max_episodes} days')
    while(validation_environment.there_is_another_episode() and (max_episodes == -1 or episode_count < max_episodes)):

        # Reset the environment and obtain the initial observation
        observation = validation_environment.reset()
        agent.reset_invetory()
        info = {}

        while True:

            action, action_by_model, action_buy_by_model, action_sell_by_model = agent.act(observation)
            actions_by_model += action_by_model
            buy_actions += action_buy_by_model
            sell_actions += action_sell_by_model
            next_observation, reward, done, info = validation_environment.step(action)
            
            if info['net_profit'] != 0:
                total_net_profit.append(info['net_profit'])
            if info['profit_loss'] != 0:
                total_profit_loss.append(info['profit_loss'])

            if done: break

            observation = next_observation

        # print(f"{np.sum(total_net_profit)} -  {np.sum(total_profit_loss)}")
        episode_count+=1
        #Ogni 250 episodi/giorni calcolo le metriche: quindi calcolo le metriche ogni anno
        if episode_count % 250 == 0:

            net_profit = np.array(total_net_profit)
            profit_loss = np.array(total_profit_loss)

            print("Happy new year")
            print(f"Net profit: {np.sum(net_profit)}")
            print(f"Profit: {np.sum(profit_loss)}")
            #Calculate Sharpe Ratio metric
            if(len(profit_loss) > 0):
                sharpeRatio = sharpe_ratio(profit_loss,risk_free=0)
                print('Sharpe Ratio: ', sharpeRatio)

                #Calculate Value at Risk metric
                valueAtRisk = value_at_risk(profit_loss)
                print('Value at Risk: ', valueAtRisk)

                #Calculate Conditional Value at Risk metric
                condValueAtRisk = conditional_value_at_risk(profit_loss)
                print('Conditional Value at Risk: ', condValueAtRisk)

            #Calculate Maximum Drawdown metric
            maxDrawdown = max_drawdown(profit_loss)
            print('Maximum Drawdown: ', maxDrawdown)

            #Calculate Compounded Annual Return metric
            annualReturn = cagr(profit_loss, annualization=1)
            print('Compounded Annual Return: ', annualReturn)

            #Calculate Annual Volatility metric
            annualVolatility = annual_volatility(profit_loss, annualization=1)
            print('Annual Volatility: ', annualVolatility)

    print(f'Buys by model {str(buy_actions)}/{str(actions_by_model)} Sells by model {str(sell_actions)}/{str(actions_by_model)}')
    print(f'>>> Validation finished! <<< \n')

# -------- Validation finished -----------

# -------- Train -----------

episode_count = 0
rewards_list = []

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
    buy_actions = 0
    sell_actions = 0

    while True:

        action, action_by_model, action_buy_by_model, action_sell_by_model = agent.act(observation)
        actions_by_model += action_by_model
        buy_actions += action_buy_by_model
        sell_actions += action_sell_by_model
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
        agent.epsilonDecay()

    if episode_count % 10:
        target_net_state_dict = agent.model.state_dict()
        policy_net_state_dict = agent.target_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        agent.target_model.load_state_dict(target_net_state_dict)

    episode_count += 1

    if episode_count % 50 == 0:
        agent.evaluation_mode()
        perform_validation(episode_count, 251)
        agent.evaluation_mode(False)

    net_profit = np.array(net_profit)
    profit_loss = np.array(profit_loss)

    print(f">>> EPISODE: {episode_count} <<<")
    print(f"Reward: {np.mean(rewards_list):3.5f}, Profits: {sum(profit_loss)}, BUY: {len(info['when_bought'])}, SELL: {len(info['when_sold'])}") #, Time: {time.time()-start_time}")
    
    print("---- Metrics ----")

    #Calculate Success Rate metric
    print(f'Buys by model {str(buy_actions)}/{str(actions_by_model)} Sells by model {str(sell_actions)}/{str(actions_by_model)}')
    print(f"Actual trades : { info['actual_trades']}")
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

agent.evaluation_mode()
perform_validation(episode_count, 750)
agent.evaluation_mode(False)