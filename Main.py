#Main File for Training. Please see the Main.ipynb for more details


import gym
import numpy as np
import matplotlib.pyplot as plt
from support import QLAgent, make_file_structure
import os
import pandas as pd
import datetime

"""1. Hyperparameters for Run"""

gammas = [0.01]
for x in range(10,1000):
    if x % 50 == 0:
        gammas.append(x/1000)
gammas.append(1)


episodes = 1500
max_steps = 200
agents_per_setting = 50


env = gym.make('Taxi-v3')
env_name = 'Taxi-v3'
alpha = 0.1
epsilon = 0.9
epsilon_min = 0.05
epsilon_decay_rate = 0.01

training_folder = ('Run3-Taxiv3-gammas') #Folder where all data will be stored
print(gammas)

"""
3. Training of Agents
"""
seeds = np.load('seeds.npy')
make_file_structure(training_folder, gammas, alpha, agents_per_setting, env_name, gammas, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps)

for gamma in gammas:
    #making new folder that will hold all the agents data files
    print(f'-----------Training GAMMA: {alpha}-----------')
    for n_agent in range (1,agents_per_setting+1):
        Agent = QLAgent(env, alpha, gamma, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps)
        return_list, reward_per_step = Agent.train()
        np.save(f'{training_folder}/agents_for_setting_{alpha}/agent_Q-Table_{n_agent}',Agent.Q)
        np.save(f'{training_folder}/agents_for_setting_{alpha}/agent_reward_per_step_{n_agent}', reward_per_step)

#Storing Date and Time of final episode to keep track of duration of one training episode and ensure completion
with open(f"{training_folder}/config.txt", "a") as file:
    file.write(f'end_of_trainingrun: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# Evaluation
for gamma in gammas:
    for n_agent in range (1,agents_per_setting+1):
        Return_array = np.array([])
        Return_array = Return_array[:,np.newaxis]
        for seed in seeds:
            Return_array = np.vstack((Return_array,Agent.evaluate(seed=seed)))
        np.save(f'{training_folder}/agents_for_setting_{alpha}/agent_evaluation_return_{n_agent}', Return_array)