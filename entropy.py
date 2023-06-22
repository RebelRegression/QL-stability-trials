import numpy as np
import scipy
import gym
import matplotlib.pyplot as plt
import os
from support import *

training_folder = ('Run6-Taxiv3-alpha-n_50')
config_dictionary = read_config_from_txt_file(f'{training_folder}/config.txt')

variable_parameter = config_dictionary.get('variable_parameter')
variable_parameter_list = config_dictionary.get(variable_parameter)

#making directories where figures will be saved
try: 
    os.mkdir(f'{training_folder}/figures')
except:
    pass
try:
    os.mkdir(f'{training_folder}/figures/entropy')
except:
    pass

for variable_parameter_value in variable_parameter_list: 
    agent_entropy_array = np.empty([500,0])

    for n_agent in range(1,config_dictionary.get('n_agents') +1):
        Q_table = np.load(f'{training_folder}/agents_for_setting_{variable_parameter_value}/agent_Q-Table_{n_agent}.npy')
        state_entropy = []

        for state in Q_table:

            # 1. calulate probabilites
            probabilites = np.exp(state)/sum(np.exp(state))
            # 2. calculate entropy with base6 for 6 actions
            state_entropy.append(scipy.stats.entropy(probabilites, base=6))


        state_entropy = np.array(state_entropy)  
        state_entropy = state_entropy[:,np.newaxis]
        agent_entropy_array = np.hstack((agent_entropy_array, state_entropy))

    np.save(f'{training_folder}/agents_for_setting_{variable_parameter_value}/entropy_array_{variable_parameter_value}.npy', agent_entropy_array)
    plt.imshow(agent_entropy_array, cmap='hot', interpolation='nearest')
    plt.title(label=f'Agent Entropy for {config_dictionary.get("variable_parameter")}: {variable_parameter_value}',pad=+10)
    plt.ylabel('State')
    plt.xlabel('Agent-ID')
    plt.colorbar(shrink=0.6, anchor=(0.0, 0.0))
    plt.savefig(f'{training_folder}/figures/entropy/entropy_heatmap_alpha{variable_parameter_value}.jpg', dpi=2000, bbox_inches='tight')
    plt.clf()
    print(f'{variable_parameter}:', '{:.2f} done'.format(variable_parameter_value))

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