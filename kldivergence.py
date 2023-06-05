import numpy as np
import scipy
import gym
import matplotlib.pyplot as plt
import os

alphas = [0.01]
for x in range(10,1000):
    if x % 50 == 0:
        alphas.append(x/1000)
alphas.append(1)

episodes = 3000
max_steps = 200
agents_per_setting = 50

env = gym.make('Taxi-v3')
env_name = 'Taxi-v3'
gamma = 0.9
epsilon = 0.9
epsilon_min = 0.05
epsilon_decay_rate = 0.01

training_folder = ('Run2-Taxiv3-alphas') #Folder where all data will be stored
print(alphas)

try: 
    os.mkdir(f'{training_folder}/figures')
except:
    pass
try:
    os.mkdir(f'{training_folder}/figures/kl-divergence')
except:
    pass


for alpha in alphas: 
    try:
        os.mkdir(f'{training_folder}/agents_for_alpha_{alpha}/temp')
    except:
        pass
    agent_entropy_array = np.empty([500,0])

    for n_agent in range(1,agents_per_setting +1):
        Q_table = np.load(f'{training_folder}/agents_for_alpha_{alpha}/agent_Q-Table_{n_agent}.npy')
        action_probabiliites = []

        for state in Q_table:

            # 1. calulate probabilites
            probabilities = np.exp(state)/sum(np.exp(state))
            # 2. calculate entropy with base6 for 6 actions
            action_probabiliites.append(probabilities)


        array = np.array(action_probabiliites)  
        np.save(f'{training_folder}/agents_for_alpha_{alpha}/temp/action_probabilites_{n_agent}', array)

# Vergleichen aller Action probabilites eines jeden Agenten mit denen von allen anderen Agenten 
for alpha in alphas:
    for agent_x in range(1,agents_per_setting +1):
        for agent_y in range(1,agent_entropy_array+1):

            if agent_x == agent_y: # Damit nicht die gleichen Action prob. distributions miteinander verglichen werden
                pass 

            action_probabiliites_agent_x = np.load(f'{training_folder}/agents_for_alpha_{alpha}/temp/action_probabilites_{agent_x}')
            action_probabiliites_agent_y = np.load(f'{training_folder}/agents_for_alpha_{alpha}/temp/action_probabilites_{agent_y}')

    """plt.imshow(agent_entropy_array, cmap='hot', interpolation='nearest')
    plt.title(f'Agent Entropy for Alpha: {alpha}')
    plt.ylabel('State')
    plt.xlabel('Agent Number')
    plt.savefig(f'{training_folder}/figures/entropy/entropy_heatmap_alpha{alpha}.jpg', dpi=2000)"""