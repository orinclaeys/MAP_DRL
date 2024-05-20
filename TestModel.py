import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from DRL import DEVICE, Agent, State

model_path = 'C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/Finished_weights/model_weights_32_AP1.pth'
agent = Agent()
agent.loadWeights(model_path)

states_path='C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/training_data/Validation/states_V4.csv'
actions_path='C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/training_data/Validation/actions_V4.csv'

with open(states_path,'r') as file:
    csvreader = csv.reader(file)
    states = list(csvreader)

with open(actions_path,'r') as file:
    csvreader = csv.reader(file)
    actions = list(csvreader)

latencies_drl = []
latencies_rule= []
latencies_best= []
power=[]
mistakes = 0
transitions = 0
for i in range(5000):
    latencies = actions[i]
    values = states[i]
    rsu_pows = [float(values[3]),float(values[9]),float(values[15])]
    #state = State(float(values[0]),float(values[1]), float(values[2]), float(values[3]),float(values[4]), float(values[5]), float(values[6]),float(values[7]), float(values[8]), float(values[9]), float(values[10]), float(values[11]), float(values[12]), float(values[13]), float(values[14]), float(values[15]), float(values[16]), float(values[17]), float(values[18]), float(values[19]), float(values[20]))
    #state = State(float(values[0]),float(values[1]), float(values[2]), float(values[3]),float(values[6]), float(values[7]), float(values[8]),float(values[9]), float(values[12]), float(values[13]), float(values[14]), float(values[15]), float(values[20]))
    state = State(float(values[0]),float(values[3]),float(values[6]),float(values[9]),float(values[12]),float(values[15]),float(values[20]))
    state = torch.tensor(state, dtype = torch.float32,  device=DEVICE).unsqueeze(0)
    
    #DRL-based Action
    action_drl = agent.select_action(state)
    action_drl = action_drl[0].item()
    #Rule-based Action
    cpus = [float(values[0]),float(values[6]),float(values[12])]
    cpus = np.array(cpus)
    action_rule = np.argmin(cpus)
    #Best Action
    latencies_np = np.array(latencies)
    action_best = np.argmin(latencies_np)

    #DRL-based Latency
    drl_latency = float(latencies[action_drl])
    #Best possible Latency
    latencies_float = np.array([float(x) for x in latencies])
    best_latency = float(min(latencies_float))
    #Rule-based Latency
    rule_latency = float(latencies[action_rule])

    #Count violations
    if drl_latency > 100:
        mistakes = mistakes + 1
    #Count transitions
    if action_drl != float(values[20]):
        transitions = transitions + 1


    latencies_drl.append(drl_latency)
    latencies_best.append(best_latency)
    latencies_rule.append(rule_latency)
    power.append(float(rsu_pows[action_rule]))

best_average = sum(latencies_best)/len(latencies_best)
drl_average  = sum(latencies_drl)/len(latencies_drl)
rule_average = sum(latencies_rule)/len(latencies_rule)
power_average = sum(power)/len(power)

print("Best average: "+str(best_average))
print("DRL average: "+str(drl_average))
print("Rule average: "+str(rule_average))
print("Mistakes: "+str(mistakes))
print("Transitions: "+str(transitions))
print("Power: " +str(power_average))

plt.plot(latencies_drl,label='DRL-based')
plt.plot(latencies_best,label='Best')
plt.plot(latencies_rule,label='Rule-based')

plt.yscale('log')
plt.show()


    