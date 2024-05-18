from agent.agent import Agent
from agent.memory import Transition, ReplayMemory
from functions import *
from tqdm import tqdm
import torch

window_size = 32
episode_count = 2500

agent = Agent(window_size)
#data = getStockDataVec('IVV_1m_training')
data = readData('data/IVV_1m_training.csv')
l = len(data) - 1
timeSeenData = 1 #numero di volte che leggo l'intero dataset

for z in range(timeSeenData):
	for e in tqdm(range(len(data))):
		print("Episode " + str(e) + "/" + str(episode_count))
		state = getState(data[e], 0, window_size + 1)

		total_profit = 0
		agent.inventory = []

		for t in tqdm(range(len(data[e])-1)):
			action = agent.act(state)

			# sit
			next_state = getState(data[e], t + 1, window_size + 1)
			reward = 0

			if action == 1: # buy
				agent.inventory.append(data[e][t])
				# print("Buy: " + formatPrice(data[t]))

			elif action == 2 and len(agent.inventory) > 0: # sell
				bought_price = agent.inventory.pop(0)
				reward = max(data[e][t] - bought_price, 0)
				total_profit += data[e][t] - bought_price
				# print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

			done = True if t == (len(data[e]) - 1) - 1 else False
			agent.memory.push(state, action, next_state, reward)
			state = next_state

			if done:
				print("--------------------------------")
				print("Total Profit: " + formatPrice(total_profit))
				print("--------------------------------")

			agent.optimize()

		if e % 10 == 0:
			agent.target_net.load_state_dict(agent.policy_net.state_dict())
		# 	torch.save(agent.policy_net, "models/policy_model")
		# 	torch.save(agent.target_net, "models/target_model")
