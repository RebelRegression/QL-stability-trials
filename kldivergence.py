import numpy as np
import scipy
import gym
import matplotlib.pyplot as plt
import os
import tqdm
from support import *



training_folder = ('Run6-Taxiv3-alpha-n_50')
config_dictionary = read_config_from_txt_file(f'{training_folder}/config.txt')


variable_parameter = config_dictionary.get('variable_parameter')
variable_parameter_list = config_dictionary.get(variable_parameter)


try: 
    os.mkdir(f'{training_folder}/figures')
except:
    pass
try: 
    os.mkdir(f'{training_folder}/stat_data')
except:
    pass
try:
    os.mkdir(f'{training_folder}/figures/kl-divergence')
except:
    pass


#Erstellen von arrays der action probabilites für alle trainierten Agenten
for variable_parameter_value in variable_parameter_list:
    for n_agent in range(1, config_dictionary.get('n_agents')+1):
        Q_table = np.load(f'{training_folder}/agents_for_setting_{variable_parameter_value}/agent_Q-Table_{n_agent}.npy')
        action_probabilities_array = get_action_probabilities_as_array(Q_table)
        np.save(f'{training_folder}/agents_for_setting_{variable_parameter_value}/agent_action_probabilities_{n_agent}.npy', action_probabilities_array)
    print('action probabilities calculeted for {:.2f}'.format(variable_parameter_value))

#Berechnen der KL Divergenz von jedem Agenten pro Setting zu jedem anderen Agenten pro Setting 

for variable_parameter_value in variable_parameter_list:

    agents_kl_divergence = np.empty([0,config_dictionary.get('n_agents')])
    for agent_x in range(1, config_dictionary.get('n_agents')+1):
        agent_x_probabilites = np.load(f'{training_folder}/agents_for_setting_{variable_parameter_value}/agent_action_probabilities_{agent_x}.npy')
        agent_x_kl_divergence_2_all_other_agents = []
        
        for agent_y in range(1, config_dictionary.get("n_agents")+1):
            #pass wenn die AgentID gleich ist, um nicht den gleichen Agenten zueinander zu vergleichen


            agent_y_probabilites = np.load(f'{training_folder}/agents_for_setting_{variable_parameter_value}/agent_action_probabilities_{agent_y}.npy')

            num_states = len(agent_x_probabilites)
            current = [] #an array that contains the kl divergence between all states of two agents

            for state in range(num_states):
                current.append(scipy.stats.entropy(agent_x_probabilites[state], agent_y_probabilites[state], base=6))
            agent_x_kl_divergence_2_all_other_agents = np.append(agent_x_kl_divergence_2_all_other_agents, [np.mean(current)])
            

        #speichert den Durchschnitt der KL-Divergenz zwischen zwei Agenten
        agents_kl_divergence = np.vstack((agents_kl_divergence, agent_x_kl_divergence_2_all_other_agents))

    np.save(f'{training_folder}/stat_data/kl-divergence_array_setting_{variable_parameter_value}.npy', agents_kl_divergence)      
    plt.imshow(agents_kl_divergence, cmap='hot', interpolation='nearest')
    plt.title(label=f'average KL-Divergence for {config_dictionary.get("variable_parameter")}: {variable_parameter_value}', pad=+10)
    plt.ylabel('Agent-Y')
    plt.xlabel('Agent-X')
    plt.colorbar(shrink=0.6, anchor=(0.0, 0.0))
    plt.savefig(f'{training_folder}/figures/kl-divergence/kl-divergence_heatmap_setting_{variable_parameter_value}.jpg', dpi=2000, bbox_inches='tight')
    plt.clf()
    print('Setting {:.2f} done'.format(variable_parameter_value))