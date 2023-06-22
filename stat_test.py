'''
Datei, welche alle statistischen Tests mit den Trainingsdaten durchführt. 
'''
from support import *
import scipy
import matplotlib.pyplot as plt



training_folder = ('Run6-Taxiv3-alpha-n_50')
config_dictionary = read_config_from_txt_file(f'{training_folder}/config.txt')


variable_parameter = config_dictionary.get('variable_parameter')
variable_parameter_list = config_dictionary.get(variable_parameter)
n_agents = config_dictionary.get('n_agents')

try: 
    os.mkdir(f'{training_folder}/figures')
except:
    pass
try:
    os.mkdir(f'{training_folder}/figures/stat_tests')
except:
    pass

# Zwei Listen, welche alle Populations parameter enthalten, bei denen die Returnverteilungen entweder normalverteilt oder nicht
# normalverteilt sind. 
norm_distr = []
not_norm_distr = []

for parameter in variable_parameter_list:
    average_return_per_setting = []
    for agent in range(1,config_dictionary.get('n_agents')+1):
        returnlist = np.load(f'{training_folder}/agents_for_setting_{parameter}/agent_evaluation_return_{agent}.npy')
        average_return_per_setting.append(returnlist.mean())
    
    calc_w, a = scipy.stats.shapiro(average_return_per_setting)
 
    # Prüfen ob alpha <= 5, wenn True dann handelt es sich wsl um normalverteilte Werte
    if a >= 0.05:
        norm_distr.append(parameter)
    else:
        not_norm_distr.append(parameter)

print('Folgende Parameter haben zu Normalverteilten Return-Verteilungen geführt: ', norm_distr)
print('Folgende Parameter haben zu anderen Return-Verteilungen geführt: ', not_norm_distr)

## Hier ebenfalls noch als heatmap alle Populationen anzeigen und farblich kodieren, welche normalverteilt sind. 

# 2. Hier alle Normalverteilten Distributionen mit f und t-Test prüfen

return_per_setting_array = []

for parameter in norm_distr:
    average_return_per_setting = []
    for agent in range(1,config_dictionary.get('n_agents')+1):
        returnlist = np.load(f'{training_folder}/agents_for_setting_{parameter}/agent_evaluation_return_{agent}.npy')
        average_return_per_setting.append(returnlist.mean())
    return_per_setting_array.append(average_return_per_setting)

#    
# 1. T-Test testen von jeder Kombination an Return-Verteilungen
# 


ttest_pos = []
ttest_neg = []
for x in return_per_setting_array:
    for j in return_per_setting_array:
        if x != j:
            U1, p_ttest = scipy.stats.ttest_ind(x,j)
            if p_ttest > 0.05:
                ttest_pos.append([norm_distr[return_per_setting_array.index(x)],norm_distr[return_per_setting_array.index(j)]])
            else:
                ttest_neg.append([norm_distr[return_per_setting_array.index(x)],norm_distr[return_per_setting_array.index(j)]])

print('\n---T-Test---')
if ttest_pos:
    print('Folgende Verteilungen haben den gleichen Mittelwert zueinander: ')
    for element in ttest_pos:
        print(element)
if ttest_neg:
    print('Folgende Verteilungen haben nicht den gleichen Mittelwert zueinander: ')
    for element in ttest_neg:
        print(element)
#  
# 2. f-Test, testen aller normalverteilten Return-Verteilungen auf die gleiche Varianz
#

ftest_pos = []
ftest_neg = []
for x in return_per_setting_array:
    for j in return_per_setting_array:
        if x != j:
            x_arr = np.array(x)
            j_arr = np.array(j)
            
            f = np.var(x_arr, ddof=1) / np.var(j_arr, ddof=1)
            nun = x_arr.size-1
            dun = j_arr.size-1
            p_ftest = 1-scipy.stats.f.cdf(f, nun, dun)
            
            if p_ftest > 0.05:
                ftest_pos.append([norm_distr[return_per_setting_array.index(x)],norm_distr[return_per_setting_array.index(j)]])
            else:
                ftest_neg.append([norm_distr[return_per_setting_array.index(x)],norm_distr[return_per_setting_array.index(j)]])
print('\n---F-Test---')
if ftest_pos:
    print('Folgende Verteilungen haben die gleiche Varianz zueinander: ')
    for element in ftest_pos:
        print(element)
if ftest_neg:
    print('\nFolgende Verteilungen haben nicht die gleiche Varianz zueinander: ')
    for element in ftest_neg:
        print(element)

#
# 3. Alle nicht normal verteilten distributionen mit Mann-Whitney-U-Test testen: 
#

## Extrahieren aller Return-Verteilungen aller Populationen, welche nicht normalverteilt sind:
return_per_setting_array = []
for parameter in not_norm_distr:
    average_return_per_setting = []
    for agent in range(1,config_dictionary.get('n_agents')+1):
        returnlist = np.load(f'{training_folder}/agents_for_setting_{parameter}/agent_evaluation_return_{agent}.npy')
        average_return_per_setting.append(returnlist.mean())
    return_per_setting_array.append(average_return_per_setting)
    
## Prüfen aller Populationen zueinander: 
mannwhitney_pos = []
mannwhitney_neg = []
mannwhitney_arr = np.zeros((len(return_per_setting_array), len(return_per_setting_array)))
for x in return_per_setting_array:
    for j in return_per_setting_array:
        #if x != j:
        U1, p_mannwhitneyu = scipy.stats.mannwhitneyu(x,j)
        if p_mannwhitneyu >= 0.05:
            mannwhitney_arr[return_per_setting_array.index(x),return_per_setting_array.index(j)] = 1
            mannwhitney_pos.append([not_norm_distr[return_per_setting_array.index(x)],not_norm_distr[return_per_setting_array.index(j)]])
        else:
            mannwhitney_neg.append([not_norm_distr[return_per_setting_array.index(x)],not_norm_distr[return_per_setting_array.index(j)]])



print('\n---Mann-Whitney-U-Test---')

# Create a colormap with red and green colors
colors = ['red', 'green']
cmap = plt.cm.colors.ListedColormap(colors)

# Set the figure size
plt.figure(figsize=(10, 10))

# Plot the array as a heatmap
plt.imshow(mannwhitney_arr, cmap=cmap)

# Get the length of the array and generate tick labels
num_rows, num_cols = mannwhitney_arr.shape
row_labels = [f'{element}' for element in not_norm_distr]
col_labels = [f'{element}' for element in not_norm_distr]

# Add tick labels to the x-axis and y-axis
plt.xticks(range(num_cols), col_labels, ha='center')
plt.yticks(range(num_rows), row_labels, va='center')

# Manually add the grid lines
plt.vlines(np.arange(num_cols) + 0.5, ymin=-0.5, ymax=num_rows - 0.5, colors='black', linewidth=0.5)
plt.hlines(np.arange(num_rows) + 0.5, xmin=-0.5, xmax=num_cols - 0.5, colors='black', linewidth=0.5)

# Defining Title
plt.title(f'MWU-Test for n={n_agents} Return-Distr \n Populations with MWU a > 0.05 are green and therefore from the same distribution')
# Display the plot

plt.savefig(f'{training_folder}/figures/stat_tests/MWU-Test_n={n_agents}.jpg', dpi=2000, bbox_inches='tight')


#
# 4. Leve Test für alle nicht normalverteilten Returnverteilungen
#


stat, p = scipy.stats.levene(*return_per_setting_array)
#stat, p = scipy.stats.levene(return_per_setting_array[4], return_per_setting_array[5])
print('\n---Levene Test---')
if p > 0.05:
    print('Die Verteilungen haben die gleiche Varianz mit p: ', p)
else:
    print('Die Verteilungen haben nicht die gleiche Varianz mit p: ', p)
    
    
levene_pos = []
levene_neg = []
levene_arr = np.zeros((len(return_per_setting_array), len(return_per_setting_array)))
for x in return_per_setting_array:
    for j in return_per_setting_array:
        #if x != j:
        U1, p_leveneu = scipy.stats.levene(x,j)
        if p_leveneu >= 0.05:
            levene_arr[return_per_setting_array.index(x),return_per_setting_array.index(j)] = 1
            levene_pos.append([not_norm_distr[return_per_setting_array.index(x)],not_norm_distr[return_per_setting_array.index(j)]])
        else:
            levene_neg.append([not_norm_distr[return_per_setting_array.index(x)],not_norm_distr[return_per_setting_array.index(j)]])


# Create a colormap with red and green colors
colors = ['red', 'green']
cmap = plt.cm.colors.ListedColormap(colors)

# Set the figure size
plt.figure(figsize=(10, 10))

# Plot the array as a heatmap
plt.imshow(levene_arr, cmap=cmap)

# Get the length of the array and generate tick labels
num_rows, num_cols = levene_arr.shape
row_labels = [f'{element}' for element in not_norm_distr]
col_labels = [f'{element}' for element in not_norm_distr]

# Add tick labels to the x-axis and y-axis
plt.xticks(range(num_cols), col_labels, ha='center')
plt.yticks(range(num_rows), row_labels, va='center')

# Manually add the grid lines
plt.vlines(np.arange(num_cols) + 0.5, ymin=-0.5, ymax=num_rows - 0.5, colors='black', linewidth=0.5)
plt.hlines(np.arange(num_rows) + 0.5, xmin=-0.5, xmax=num_cols - 0.5, colors='black', linewidth=0.5)

# Defining Title
plt.title(f'Levene-Test for n={n_agents} Return-Distr \n Populations with MWU a > 0.05 are green and have therefore the same Varianz')
# Display the plot
plt.savefig(f'{training_folder}/figures/stat_tests/Levene-Test_n={n_agents}.jpg', dpi=2000, bbox_inches='tight')


#
# 5. Heatmap für Verteilungen die MWU und Lavene positiv sind
#

result_arr = np.logical_and(levene_arr, mannwhitney_arr).astype(int)

# Create a colormap with red and green colors
colors = ['red', 'green']
cmap = plt.cm.colors.ListedColormap(colors)

# Set the figure size
plt.figure(figsize=(10, 10))

# Plot the array as a heatmap
plt.imshow(result_arr, cmap=cmap)

# Get the length of the array and generate tick labels
num_rows, num_cols = result_arr.shape
row_labels = [f'{element}' for element in not_norm_distr]
col_labels = [f'{element}' for element in not_norm_distr]

# Add tick labels to the x-axis and y-axis
plt.xticks(range(num_cols), col_labels, ha='center')
plt.yticks(range(num_rows), row_labels, va='center')

# Manually add the grid lines
plt.vlines(np.arange(num_cols) + 0.5, ymin=-0.5, ymax=num_rows - 0.5, colors='black', linewidth=0.5)
plt.hlines(np.arange(num_rows) + 0.5, xmin=-0.5, xmax=num_cols - 0.5, colors='black', linewidth=0.5)

# Defining Title
plt.title('Populations that pass both the Levene and MWU-Test are shown in green')
# Display the plot
plt.savefig(f'{training_folder}/figures/stat_tests/MWU_and_Levene_n={n_agents}.jpg', dpi=2000, bbox_inches='tight')
