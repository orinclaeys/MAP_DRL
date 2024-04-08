from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import math
import re
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cpu")
# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01
LR = 1e-4
MEMORYCAPACITY = 2000
EPISODES = 30
# Environment parameters
action_space = (0,1,2)
n_actions = len(action_space)
n_observations = 7
latency_desired = 100

# Transition class for storing transitions in the replay-memory
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# State class for storing a single state, includes CPU, memory and power usage of three RSU's and the latency from the OBU
#State = namedtuple('State',('RSU1_cpu','RSU1_memory','RSU1_power', 'RSU2_cpu', 'RSU2_memory', 'RSU2_power', 'RSU3_cpu', 'RSU3_memory', 'RSU3_power', 'OBU_latency'))
State = namedtuple('State',('RSU1_cpu','RSU1_power', 'RSU2_cpu', 'RSU2_power', 'RSU3_cpu','RSU3_power','OBU_latency'))

# Functions for reward & punishment calculation
def calcReward1(latency):
    """Implements the first reward function (see docs)"""
    if latency < latency_desired:
        reward = 5000
    else:
        reward = latency_desired-latency
    return reward

def calcReward2(oldLatency, newLatency, oldApplication, newApplication):
    """Implements the second reward function (see docs)"""
    reward = oldLatency - newLatency
    if oldLatency < latency_desired and oldApplication!=newApplication:
        reward = -50
    return reward

# ReplayMemory class
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Take a random sample of batch_size from the replay memory"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return the length of the memory"""
        return len(self.memory)

# Class for the Q-neural network    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)
    
    def forward(self, x):
        """Forwarding a tensor X through the neural network"""
        """Can be called with one element to determine next action or a batch during optimization"""
        """Returns tensor"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Tracker():

    def __init__(self):
        self.epsilonT = []
        self.epsilonValues = []
        self.rewardsT = []
        self.rewardsValue = []
        self.average = deque([],250)
    
    def addEpsilon(self,t, epsilon):
        """Add an epsilon value to the tracker"""
        self.epsilonT.append(t)
        self.epsilonValues.append(epsilon)
    
    def addReward(self,t,reward):
        """Add a reward value to the tracker"""
        self.rewardsT.append(t)
        self.average.append(reward)
        self.rewardsValue.append(sum(self.average)/len(self.average))
    
    def getAverageReward(self):
        """Get the latest average reward value"""
        if len(self.rewardsValue) < 1:
            print("length is zero")
            return 0
        else:
            return self.rewardsValue[-1]
    
    def showEpsilon(self):
        """Plot epsilon vs training steps"""
        plt.plot(self.epsilonT,self.epsilonValues)
        plt.xlabel('Training steps')
        plt.ylabel('Epsilon')
        plt.title('Epsilon')
        plt.show()

    def showRewards(self):
        """Plot average reward vs training steps"""
        plt.plot(self.rewardsT,self.rewardsValue)
        plt.xlabel('Training steps')
        plt.ylabel('Average reward')
        plt.title('Rewards')
        plt.show()
    
    def showBoth(self):
        """Plots both average reward and epsilon vs training steps"""
        fig, ax1 = plt.subplots()
        ax1.plot(self.epsilonT,self.epsilonValues, color='tab:blue', label='Epsilon')
        ax1.set_xlabel('Training steps')
        ax1.set_ylabel('Epsilon', color='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(self.rewardsT,self.rewardsValue, color='tab:red', label='Average Reward')
        ax2.set_ylabel('Average Reward', color='tab:red')

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc='upper right')

        plt.title('Epsilon and Reward vs Training steps')
        plt.show()
    
    def calcEpsilon(self,t,adaptive = False):
        """Calculates the epsilon value, can be either exponential(default) or adaptive"""
        if adaptive:
            epsilon = max(min((150 - abs(self.getAverageReward()+50))/100,0.9),0.1)
            print(epsilon)
            return epsilon
        else:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * t / EPS_DECAY)
            return eps_threshold


# Class to define the actual agent
class Agent(object):

    def __init__(self):
        self.policy_net = DQN(n_observations, n_actions).to(DEVICE)
        self.target_net = DQN(n_observations, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORYCAPACITY)
        self.steps_done = 0
        self.current = 0
        self.tracker = Tracker()

    
    # Method to chose an action base on a the current state as input. The action can be random or calculated by the policy_network based on epsilon
    def select_action(self, state, training=False):
        if training is True:
            sample = random.random()
            eps_threshold = self.tracker.calcEpsilon(self.steps_done)
            self.steps_done += 1
            self.tracker.addEpsilon(self.steps_done,eps_threshold)
            #print("step:" + str(self.steps_done)+ "->" + "epsilon: " + str(eps_threshold))

            if sample > eps_threshold:
                own_decision = True
                with torch.no_grad():
                    action = self.policy_net(state).max(1).indices.view(1,1)
                    return action, own_decision
            else:
                own_decision = False
                action = torch.tensor([[random.choice(action_space)]], device=DEVICE, dtype=torch.long)
                return action, own_decision
        else:
            action = self.policy_net(state).max(1).indices.view(1,1)
            return action, True

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

        #print(self.policy_net(state_batch))
        #print(action_batch)
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
    def getState(self, i):
        with open('states.csv', 'r') as statefile:
            csvreader = csv.reader(statefile)
            rows = list(csvreader)
            values = rows[i]

        
        state = State(float(values[0]), float(values[2]), float(values[3]), float(values[5]), float(values[6]), float(values[8]),float(values[9]))
        return state
    
    # Method to perform an action
    def doAction(self, action,i):
        with open('actions.csv') as actionfile:
            csvreader = csv.reader(actionfile)
            rows = list(csvreader)
            values = rows[i]
        if action[0].item() == 0:
            latency = float(values[0])
        elif action[0].item() == 1:
            latency = float(values[1])
        else:
            latency = float(values[2])
        state = self.getState(i)
        #new_state = State(state.RSU1_cpu,state.RSU1_memory,state.RSU1_power,state.RSU2_cpu,state.RSU2_memory,state.RSU2_power,state.RSU3_cpu,state.RSU3_memory,state.RSU3_power,latency)
        new_state = State(state.RSU1_cpu,state.RSU1_power,state.RSU2_cpu,state.RSU2_power,state.RSU3_cpu,state.RSU3_power,latency)
        reward = calcReward1(latency)
        return new_state, reward

    # Method for training the Agent
    def train(self):

        for episode in range(EPISODES):
            wrong = 0
            own_decisions = 0
            actions = [0,0,0]
            for i in range(5000):
                state = self.getState(i)
                
                state = torch.tensor(state, dtype = torch.float32,  device=DEVICE).unsqueeze(0)
                action, own_decision = self.select_action(state, True)
                
                
                observation, reward = self.doAction(action,i)
                
                if reward < 0:
                    wrong = wrong + 1
                if own_decision:
                    own_decisions = own_decisions + 1
                    #print(action[0].item())
                if action[0].item() == 0:
                    actions[0] = actions[0]+1
                if action[0].item() == 1:
                    actions[1] = actions[1]+1
                if action[0].item() == 2:
                    actions[2] = actions[2]+1
                reward = torch.tensor([reward], device=DEVICE)

                if own_decision:
                    self.tracker.addReward(self.steps_done,reward.item())


                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store transition in memory
                self.memory.push(state, action, next_state, reward)

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                time.sleep(0.5)
                
            with open('results.txt','a') as file:
                file.write("Episode "+str(episode)+" -> "+str(wrong)+", "+str(own_decision))
                file.write(actions)
            print("Episode "+str(episode)+" -> "+str(wrong)+","+str(own_decisions))
            print(actions)
        
        # After training, save the weights in a file
        self.saveWeights()

    # Saving the weights of the neural network
    def saveWeights(self):
        torch.save(self.policy_net.state_dict(), 'model_weights.pth')

    # Loading the weights from a file
    def loadWeights(self):
        self.policy_net.load_state_dict(torch.load('model_weights.pth'))
        self.target_net.load_state_dict(torch.load('model_weights.pth'))

    def plot(self):
        self.tracker.showBoth()
        


test_agent = Agent()
test_agent.train()







    
