import torch, torch.nn as nn
from lib.AgentNetworks import AgentLSTMNetwork
import random
import numpy as np
from collections import deque


device = "cpu" if not torch.cuda.is_available() else 'cuda'

class DQNAgent():
    def __init__(self, feature_size, window_size, is_eval=False, model_name=""):
        super(DQNAgent, self).__init__()
        self.feature_size = feature_size
        self.window_size = window_size
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.5

        self.model = AgentLSTMNetwork(self.feature_size, self.window_size, self.action_size, device, is_eval)
        self.model.to(device)
        self.target_model = AgentLSTMNetwork(self.feature_size, self.window_size, self.action_size, device, is_eval)
        self.target_model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, amsgrad=True)

        self.agent_inventory = []

    def reset_invetory(self):
        self.agent_inventory = []

    def evaluation_mode(self, eval=True):
        self.is_eval= eval

    def epsilonDecay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state): 
        #If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
        #actions suggested.

        action_selected = 0
        self.model.eval()
        action_by_model = 0
        buys = 0
        sells = 0
        # Epsilon-greedy policy only on training
        if not self.is_eval and random.random() <= self.epsilon:
            action_selected = random.randrange(self.action_size) 
        else:
            with torch.no_grad():
                state = torch.transpose(state, 0, 1).to(device)
                options = self.model(state.float(), 0).reshape(-1).cpu().numpy()   
                action_selected = np.argmax(options)
                buys = 1 if action_selected == 1 else 0
                sells = 1 if action_selected == 2 else 0
                action_by_model = 1

        # Force correct action only in evaluation mode
        # if self.is_eval:
        #     if action_selected != 0 and len(self.agent_inventory) > 0:
        #         last_action = self.agent_inventory.pop(0)
        #         if last_action == action_selected: # if BUY BUY, or SELL SELL, the action is HOLD
        #             action_selected = 0
        #             self.agent_inventory.append(last_action)
        #     elif action_selected != 0:
        #         self.agent_inventory.append(action_selected)
        
        return action_selected, action_by_model, buys, sells

    def expReplay(self, batch_size, times_shuffle=1):
        loss = nn.SmoothL1Loss()
        self.model.train()
        self.target_model.eval()

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
                next_state_values[non_final_mask] = self.target_model(non_final_next_states.float(), non_final_next_states.shape[0]).max(1).values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            output = loss(expected_state_action_values.float(), state_action_values.float())
            
            self.optimizer.zero_grad()
            output.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

        exp_repl_mean_loss /= batch_size * times_shuffle

        return exp_repl_mean_loss