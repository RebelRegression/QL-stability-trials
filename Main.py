#Main File for Training. Please see the Main.ipynb for more details


import gym
import numpy as np
import matplotlib.pyplot as plt
from support import *
import os
import pandas as pd
import datetime

"""1. Hyperparameters for Run"""
variable_parameter = 'alpha'
parameter_list = [0.01]
for x in range(10,1000):
    if x % 50 == 0:
        parameter_list.append(x/1000)
parameter_list.append(1)
subdirectory_list = parameter_list
variable_parameter_list = parameter_list

episodes = 1500
max_steps = 200
agents_per_setting = 50


env = gym.make('Taxi-v3')
env_name = 'Taxi-v3'
alpha = variable_parameter_list
gamma = 0.9
epsilon = 0.9
epsilon_min = 0.05
epsilon_decay_rate = 0.01

training_folder = ('Run6-Taxiv3-alpha-n_50') #Folder where all data will be stored
log_reward_for_every_step = False #If True a np file will be created for each agent with the reward at every step and the return per episode in the last column. 
print(variable_parameter_list)


"""
3. Training of Agents
"""
seeds = np.load('seeds.npy')
make_file_structure(training_folder, subdirectory_list, variable_parameter, alpha, agents_per_setting, env_name, gamma, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps)

for parameter in variable_parameter_list:
    #making new folder that will hold all the agents data files
    print(f'-----------Training {variable_parameter}: {parameter}-----------')
    for n_agent in range (1,agents_per_setting+1):
        Agent = QLAgent(env, parameter, gamma, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps)
        return_list, reward_per_step = Agent.train(log_reward_for_every_step=log_reward_for_every_step)
        np.save(f'{training_folder}/agents_for_setting_{parameter}/agent_Q-Table_{n_agent}',Agent.Q)
        if log_reward_for_every_step:
            np.save(f'{training_folder}/agents_for_setting_{parameter}/agent_reward_per_step_{n_agent}', reward_per_step)

        Return_array = np.array([])
        Return_array = Return_array[:,np.newaxis]
        for seed in seeds:
            Return_array = np.vstack((Return_array,Agent.evaluate(seed=seed)))
        np.save(f'{training_folder}/agents_for_setting_{parameter}/agent_evaluation_return_{n_agent}', Return_array)


#Storing Date and Time of final episode to keep track of duration of one training episode and ensure completion
with open(f"{training_folder}/config.txt", "a") as file:
    file.write(f'end_of_trainingrun; {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')