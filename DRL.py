from collections import namedtuple, deque
from random import random
from itertools import count

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.deivce("cpu")
# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORYCAPACITY = 2000
EPISODES = 200

# Environment parameters
n_actions = 3
n_observations = 10

# Transition class for storing transitions in the replay-memory
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# State class for storing a single state, includes CPU, memory and power usage of three RSU's and the latency from the OBU
State = namedtuple('State',('RSU1_cpu','RSU1_memory','RSU1_power', 'RSU2_cpu', 'RSU2_memory', 'RSU2_power', 'RSU3_cpu', 'RSU3_memory', 'RSU3_power', 'OBU_latency'))

# ReplayMemory class
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """ Take a random sample of batch_size from the replay memory"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """ Return the length of the memory"""
        return len(self.memory)

# Class for the Q-neural network    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 == nn.Linear(n_observations, 32)
        self.layer2 == nn.Linear(32, 32)
        self.layer3 == nn.linear(32, n_actions)
    
    def forward(self, x):
        """ Forwarding a tensor X through the neural network"""
        """ Can be called with one element to determine next action or a batch during optimization"""
        """ Returns tensor"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# Class to define the actual agent
class Agent(object):

    def __init__(self):
        self.policy_net = DQN(n_observations, n_actions).to(DEVICE)
        self.target_net = DQN(n_observations, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORYCAPACITY)
        self.steps_done = 0
    
    # Method to chose an action base on a the current state as input. The action can be random or calculated by the policy_network based on epsilon
    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1,1)
        else:
            return torch.tensor([[random.randint(1, n_actions)]], device=DEVICE, dtype=torch.long)

    # Method to optimize the model    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # Method to obtain the state of the environment
    def getState(self):
        return
    
    #Method to perform an action
    def doAction(self, action):
        return

    # Method for training the Agent
    def train(self):
        for i_episode in range(EPISODES):
            for t in count():
                state = self.getState()
                state = torch.tensor(state, dtype = torch.float32,  device=DEVICE).unsqueeze(0)
                action = self.select_action(state)
                observation, reward = self.doAction(action)
                reward = torch.tensor([reward], device=DEVICE)

                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store transition in memory
                self.memory.push(state, action, next_state, reward)

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()



    