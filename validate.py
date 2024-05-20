import random
import os
import numpy as np
import torch
from lib.DQNAgent import DQNAgent
from lib.IVVEnvironment import IVVEnvironment

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

window_size = 32
batch_size = 64
feature_size = 4
seed = 9
times_update_dqn = 1
TAU = 0.005

seed_everything(9)
device = "cpu" if not torch.cuda.is_available() else 'cuda'

validation_path = "lib/data/IVV_1m_validation.csv"
validation_environment = IVVEnvironment(validation_path, seed=seed, device=device, window_size=window_size, trading_cost=1e-3)
agent = DQNAgent(feature_size, window_size)
agent.load_policy('./models/')

def perform_validation(current_episode, max_episodes=-1):
    validation_environment.close()
    
    total_net_profit = []
    total_profit_loss = []
    annual_net_profit = []
    annual_profit_loss = []
    episode_count = 0
    actions_by_model = 0
    buy_actions = 0
    sell_actions = 0
    actual_trades = []
    
    print(f'Start validation: model from ep {current_episode} on {max_episodes} days')
    agent.evaluation_mode(True)  # Ensure the agent is in evaluation mode
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
                annual_net_profit.append(info['net_profit'])
            if info['profit_loss'] != 0:
                total_profit_loss.append(info['profit_loss'])
                annual_profit_loss.append(info['profit_loss'])

            if done: break

            observation = next_observation

        episode_count += 1
        actual_trades.append(info["actual_trades"])
        if episode_count % 250 == 0:

            net_profit = np.array(annual_net_profit)
            profit_loss = np.array(annual_profit_loss)

            print("Happy new year")
            print(f"Net profit: {np.sum(net_profit)}")
            print(f"Profit: {np.sum(profit_loss)}")
            #Calculate Sharpe Ratio metric
            if(len(profit_loss) > 0):
                sharpeRatio = sharpe_ratio(profit_loss, risk_free=0)
                print('Sharpe Ratio: ', sharpeRatio)

                valueAtRisk = value_at_risk(profit_loss)
                print('Value at Risk: ', valueAtRisk)

                condValueAtRisk = conditional_value_at_risk(profit_loss)
                print('Conditional Value at Risk: ', condValueAtRisk)

            maxDrawdown = max_drawdown(profit_loss)
            print('Maximum Drawdown: ', maxDrawdown)

            annualReturn = cagr(profit_loss, annualization=1)
            print('Compounded Annual Return: ', annualReturn)

            annualVolatility = annual_volatility(profit_loss, annualization=1)
            print('Annual Volatility: ', annualVolatility)

            annual_net_profit = []
            annual_profit_loss = []
            
    print(f"{np.mean(actual_trades)} trades/day")
    print(f'Buys by model {str(buy_actions)}/{str(actions_by_model)} Sells by model {str(sell_actions)}/{str(actions_by_model)}')
    print(f'>>> Validation finished! <<< \n')