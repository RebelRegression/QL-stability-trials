from support import *
import matplotlib.pyplot as plt

training_folder = 'Run4-Taxiv3-alphas'

try: 
    os.mkdir(f'{training_folder}/figures')
except:
    pass
try:
    os.mkdir(f'{training_folder}/figures/average_return')
except:
    pass

config_dictionary = read_config_from_txt_file(f'{training_folder}/config.txt')


variable_parameter = config_dictionary.get('variable_parameter')
variable_parameter_list = config_dictionary.get(variable_parameter)


for variable_parameter_value in variable_parameter_list:

    average_return_per_setting = []

    for agent in range(1,config_dictionary.get('n_agents')+1):

        returnlist = np.load(f'{training_folder}/agents_for_setting_{variable_parameter_value}/agent_evaluation_return_{agent}.npy')
        average_return_per_setting.append(returnlist.mean())
        
    average = np.mean(average_return_per_setting)
        
    # Generate x-values as integers in the range of the length of the list
    AgentIDs = range(1,len(average_return_per_setting)+1)

    # Create the bar chart
    plt.bar(AgentIDs, average_return_per_setting, color='blue')

    # Set the labels and title
    plt.xlabel('Agent IDs')
    plt.ylabel('Average Return')
    plt.title(f'Average Return for {variable_parameter}: {variable_parameter_value}')
    plt.text(len(average_return_per_setting)-1, -1, f'Average: {average:.2f}', ha='right', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    # save the plot
    plt.savefig(f'{training_folder}/figures/average_return/average_return_{variable_parameter_value}.jpg', dpi=2000, bbox_inches='tight')