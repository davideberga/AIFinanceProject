from lib.IVVEnvironmentContinue import IVVEnvironmentContinue
import numpy as np
import torch
import time
import asyncio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.a2c import A2CAgent, MemoryBuffer

device = "cpu" if not torch.cuda.is_available() else 'cuda'
import warnings
warnings.filterwarnings('ignore')


train_path = "lib/data/IVV_1m_training.csv"
validation_path = "lib/data/IVV_1m_validation.csv"

async def print_async(*args):
    print(*args)

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
batch_size = 32
feature_size = 2
seed = 9
seed_everything(9)

validation_environment = IVVEnvironmentContinue(validation_path, seed=seed, device=device, trading_cost=1e-3)

def perform_validation(agent: A2CAgent, current_episode, max_episodes=-1):
    validation_environment.close()

    total_profit_loss = 0
    episode_count = 0
    agent.reset_inventory()
    print(f'Start validation: model from ep {current_episode} on {max_episodes} days')
    while(validation_environment.there_is_another_episode() and (max_episodes == -1 or episode_count < max_episodes)):

        # Reset the environment and obtain the initial observation
        observation = validation_environment.reset()
        info = {}

        while True:

            action = agent.get_exploitation_action(observation)
            next_observation, reward, done, info = validation_environment.step(action)

            total_profit_loss += info['net_profit']

            if done: break

            observation = next_observation

        episode_count+=1

    print(f'Validation finished! Val profit, Model from ep {current_episode} on {episode_count} days: {total_profit_loss}')

MAX_BUFFER=10000
ram = MemoryBuffer(MAX_BUFFER)
agent = A2CAgent(feature_size, action_dim=1, action_lim=10, ram=ram, batch_size=0)
train_environment = IVVEnvironmentContinue(train_path, seed=seed, device=device, trading_cost=1e-3)

episode_count = 0
rewards_list = []
profit_loss_list = []

while(train_environment.there_is_another_episode()):
    # print("Running episode " + str(episode_count) + "/" + str(train_environment.num_of_ep()))

    # Reset the environment and obtain the initial observation
    observation = train_environment.reset()
    info = {}
    net_profit = []


    episode_loss = 0
    start_time = time.time()
    agent.reset_inventory()
    while True:

        action = agent.get_exploitation_action(observation)
        next_observation, reward, done, info = train_environment.step(action)
        net_profit.append(info["net_profit"])

        ram.add(observation, action, reward, next_observation)
        rewards_list.append(reward)

        if done: break

        agent.optimize()

        observation = next_observation

    episode_count += 1

    if episode_count % 20 == 0:
        pass # perform_validation(agent, episode_count, 285)
        # freezed_agent = copy.deepcopy(agent)
        # curr_val_thread = threading.Thread(target=perform_validation, args=(freezed_agent, episode_count, 285), daemon=True)
        # curr_val_thread.start()
        

    # plot_behavior(day_episode, states_buy, states_sell, total_profit)
    # print(f" >>> Episode: {episode_count} Reward: {np.mean(rewards_list):3.5f} Loss: {str(episode_loss)}, \n >>> Profit: {info['total_profit']}, BUY trades: {len(info['when_bought'])}, SELL trades: {len(info['when_sold'])}, \n >>> Time : {str(time.time() - start_time)}")
    # print(info['buy_sell_order'])

    #Calculate Success Rate metric
    successRate = info["positive_trades"]/len(info["when_sold"]) if len(info["when_sold"]) != 0  else 0
    asyncio.run(print_async('Success Rate: ', successRate))

    #Calculate Net Return metric
    netReturn = sum(net_profit)/len(net_profit)
    asyncio.run(print_async('Net Return: ', netReturn))

    #Calculate Sharpe Ratio metric
    net_profit = np.array(net_profit)
    sharpeRatio = (np.mean(net_profit) / np.std(net_profit))# - risk_free_rate
    asyncio.run(print_async('Sharpe Ratio: ', sharpeRatio))

    #Calculate Maximum Drawdown
    rollingMax = torch.cummax(torch.tensor(net_profit), 0).values
    max_drawdown = torch.tensor(net_profit) - rollingMax
    max_drawdown_percentage = (max_drawdown / rollingMax).mean()
    asyncio.run(print_async('Maximum Drawdown percentage: ', max_drawdown_percentage.item(), '%'))
    
    asyncio.run(print_async(f" >>> Episode: {episode_count} Reward: {np.mean(rewards_list):3.5f} BUY trades: {len(info['when_bought'])}, SELL trades: {len(info['when_sold'])}, Time: {time.time()-start_time}"))


