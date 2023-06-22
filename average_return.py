from support import *
import matplotlib.pyplot as plt

training_folder = 'Run6-Taxiv3-alpha-n_50'

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

# 1. Average Return for each setting figure + Data

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
    plt.clf()
    print(f'{variable_parameter}:', '{:.2f} done'.format(variable_parameter_value))


# 2. Std for each setting and figure

average_of_populations = []
std_of_populations = []

for parameter in variable_parameter_list:
    average_return_per_setting = []
    for agent in range(1,config_dictionary.get('n_agents')+1):
        returnlist = np.load(f'{training_folder}/agents_for_setting_{parameter}/agent_evaluation_return_{agent}.npy')
        average_return_per_setting.append(returnlist.mean())
        
    average_of_populations.append(np.mean(average_return_per_setting))
    std_of_populations.append(np.std(average_return_per_setting))

std_of_populations_form = [ '%.3f' % elem for elem in std_of_populations]
x_values = [var for var in variable_parameter_list]
# Convert the list to a 2D array
std_of_populations_arr = np.array(std_of_populations).reshape(1, -1)

fig, ax = plt.subplots(figsize=(15, 15))  # Adjust the figsize as per your requirement

# Plot the heatmap
heatmap = ax.imshow(std_of_populations_arr, cmap='cool', interpolation='nearest', aspect='equal')

# Set the tick positions and labels for the x-axis
ax.set_xticks(np.arange(len(x_values)))
ax.set_xticklabels(x_values)

# Set the tick positions and labels for the y-axis
ax.set_yticks([0])
ax.set_yticklabels(["Std"])
plt.xlabel('Alpha')

# Add the value annotations to each cell
for i in range(len(std_of_populations_form)):
    ax.text(i, 0, std_of_populations_form[i], ha='center', va='center', color='black')
    
# Draw black lines to separate each individual box
for i in range(std_of_populations_arr.shape[1]):
    ax.vlines(i + 0.5, -0.5, -0.5 + std_of_populations_arr.shape[0], colors='black', linewidth=1)
for j in range(std_of_populations_arr.shape[0] + 1):
    ax.hlines(j - 0.5, -0.5, -0.5 + std_of_populations_arr.shape[1], colors='black', linewidth=1)
    
#plt.colorbar(heatmap, shrink=0.6, anchor=(0.0, 0.0))
num_agents = config_dictionary.get('n_agents')
plt.title(f'Std of average return for population n={num_agents}', pad=10)

plt.savefig(f'{training_folder}/figures/average_return/00_std_return_distr_{training_folder}.jpg', dpi=2000, bbox_inches='tight')

