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

training_folder = ('Run3-Taxiv3-gammas') #Folder where all data will be stored
print(alphas)


alphas = [0.01]
for x in range(10,1000):
    if x % 50 == 0:
        alphas.append(x/1000)
alphas.append(1)

#making directories where figures will be saved
try: 
    os.mkdir(f'{training_folder}/figures')
    os.mkdir(f'{training_folder}/figures/entropy')
except:
    pass


for alpha in alphas: 
    agent_entropy_array = np.empty([500,0])

    for n_agent in range(1,agents_per_setting +1):
        Q_table = np.load(f'{training_folder}/agents_for_alpha_{alpha}/agent_Q-Table_{n_agent}.npy')
        state_entropy = []

        for state in Q_table:

            # 1. calulate probabilites
            probabilites = np.exp(state)/sum(np.exp(state))
            # 2. calculate entropy with base6 for 6 actions
            state_entropy.append(scipy.stats.entropy(probabilites, base=6))


        state_entropy = np.array(state_entropy)  
        state_entropy = state_entropy[:,np.newaxis]
        agent_entropy_array = np.hstack((agent_entropy_array, state_entropy))

    plt.imshow(agent_entropy_array, cmap='hot', interpolation='nearest')
    plt.title(f'Agent Entropy for Alpha: {alpha}')
    plt.ylabel('State')
    plt.xlabel('Agent Number')
    plt.savefig(f'{training_folder}/figures/entropy/entropy_heatmap_alpha{alpha}.jpg', dpi=2000)

#Code for splitting figures up in single parts; PROBLEM: The color scale is all messed up from image to image because they are not identical (e.g: 0 is white and in the next image its red)
"""for alpha in alphas:
    os.mkdir(f'{training_folder}/figures/entropy/alpha{alpha}')
    state_index = 100
    for i in range(5):
        agent_entropy_array = np.empty([100,0])
        for n_agent in range(1,agents_per_setting +1):
            Q_table = np.load(f'{training_folder}/agents_for_alpha_{alpha}/agent_Q-Table_{n_agent}.npy')
            state_entropy = []

            for state in Q_table[state_index-100:state_index]:

                # 1. calulate probabilites
                probabilites = np.exp(state)/sum(np.exp(state))
                # 2. calculate entropy with base6 for 6 actions
                state_entropy.append(scipy.stats.entropy(probabilites, base=6))


            state_entropy = np.array(state_entropy)  
            state_entropy = state_entropy[:,np.newaxis]
            agent_entropy_array = np.hstack((agent_entropy_array, state_entropy))


        plt.imshow(agent_entropy_array, cmap='hot', interpolation='nearest')
        plt.title(f'Agent Entropy for states {state_index-100}-{state_index}')
        plt.ylabel('states')
        plt.xlabel('agent_id')
        plt.savefig(f'{training_folder}/figures/entropy/alpha{alpha}/entropy_heatmap_alpha{alpha}-{state_index}.jpg', dpi=500)
        state_index += 100"""