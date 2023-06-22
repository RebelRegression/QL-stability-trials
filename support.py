import numpy as np
from tqdm import tqdm
import os
import gym
import pandas as pd
import datetime
import time
from IPython.display import clear_output



class QLAgent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.max_steps = max_steps
        
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        
        self.Q = np.zeros((self.n_states, self.n_actions))

    def update_Q(self,state,next_state,action,reward):
        self.Q[state,action] += self.alpha*(reward+self.gamma*self.Q[next_state].max()-self.Q[state,action])

    def softmax(self, action_values):
        e_x = np.exp(action_values - np.max(action_values))
        return e_x / e_x.sum(axis=0)

    def choose_action_softmax(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action_values = self.Q[state]
            action_probabilities = self.softmax(action_values)
            action = np.random.choice(np.arange(self.n_actions), p=action_probabilities)
        return action

    def train(self, log_reward_for_every_step=False, render=False):
        ''' Trains the Agent on the params given in the Class definition. Returns a list with the return for each episode
            as well as an array that contains the rewards for each episode and each step. The last column of the array is the return for this epsiode. 
            This is made for easy transformation to a pandas dataframe that should use column names in range n_steps and the last column name being 
            episode_return'''

        arr = np.array([])
        return_list = []

        for i in tqdm(range(self.episodes)):

            done = False
            state = self.env.reset()
            Return = 0
            reward_list = []
            for s in range(self.max_steps):

                action = self.choose_action_softmax(state)
                next_state, reward, done,_ = self.env.step(action)

                if render:
                    self.env.render()
                    time.sleep(0.5)
                    clear_output(wait=True)
                
                if log_reward_for_every_step:
                    reward_list.append(reward)
                    
                Return += reward
                self.update_Q(state,next_state,action,reward)
                state = next_state

                if done:
                    reward_list = reward_list + [None] * (self.max_steps - len(reward_list))
                    break

            if log_reward_for_every_step:   
                if arr.size == 0:
                    arr = np.append(arr, reward_list)
                else: 
                    arr = np.vstack((arr, reward_list))
            
            return_list.append(Return)
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon * self.epsilon_decay_rate)

        return_array = np.array(return_list)
        return_array = return_array[:,np.newaxis]

        if log_reward_for_every_step:
            arr  = np.hstack((arr, return_array))
        else: 
            arr = None

        return return_list, arr
    
    def evaluate(self, seed=None, render=False):

        if bool(seed):
            seed = self.env.seed(int(seed))
        else:
            self.env.seed(np.random.randint(1,10000))

        state = self.env.reset()
        Return = 0
        done = False
        for s in range(self.max_steps):
            action = np.argmax(self.Q[state])
            next_state, reward, done,_ = self.env.step(action)
            if render:
                self.env.render()
                time.sleep(0.5)
                clear_output(wait=True)
            Return += reward
            state = next_state
            if done:
                break
        
        return Return
    
    def get_action_probabilities_as_array_from_agent(self):
        '''
        Returns the action probabilities from any given QL Agent, as an array. Uses the current Q-Table as input.
        '''
        Q_table = self.Q
        action_probabiliites = []

        for state in Q_table:

            # 1. calulate probabilites
            probabilities = np.exp(state)/sum(np.exp(state))
            # 2. calculate entropy with base6 for 6 actions
            action_probabiliites.append(probabilities)

        array = np.array(action_probabiliites)  

        return array

def get_action_probabilities_as_array(Q_table):
    '''
    Returns the action probabilities from any given QL Agent, as an array. Uses the current Q-Table as input.
    '''
    action_probabiliites = []

    for state in Q_table:

        # 1. calulate probabilites
        probabilities = np.exp(state)/sum(np.exp(state))
        # 2. calculate entropy with base6 for 6 actions
        action_probabiliites.append(probabilities)

    array = np.array(action_probabiliites)  

    return array
    
def make_file_structure(training_folder, subdirectory_list, variable_parameter,alpha, agents_per_setting, env_name, gamma, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps):
    '''
    takes the parameters and creates a full subdirectory that is filled with the training data.
    '''
    
    os.mkdir(f'{training_folder}')
    make_config_txt(training_folder, variable_parameter, alpha, agents_per_setting, env_name, gamma, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps)
    for variable in subdirectory_list:
        os.mkdir(f'{training_folder}/agents_for_setting_{variable}')
    
    os.mkdir(f'{training_folder}/tmp')
    os.mkdir(f'{training_folder}/figures')

    print('Training folder created\n')

    
def make_config_txt(directory, variable_parameter, alphas, n_agents, env, gamma, epsilon, epsilon_decay_rate, epsilon_min, episodes, max_steps):
    filename = 'config.txt'
    config = {
        'variable_parameter': variable_parameter,
        'alpha': alphas,
        'n_agents': n_agents,
        'env': env,
        'gamma': gamma,
        'epsilon': epsilon,
        'epsilon_decay_rate': epsilon_decay_rate,
        'epsilon_min': epsilon_min,
        'episodes': episodes,
        'max_steps': max_steps,
        'start_of_trainingrun': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Open the file in write mode and write the dictionary
    with open(f"{directory}/{filename}", "w") as file:
        for key, value in config.items():
            file.write(f"{key}; {value}\n")

def read_config_from_txt_file(path_to_config_txt):

    # Initialize an empty dictionary
    data = {}

    # Open the text file and read its contents
    with open(path_to_config_txt, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line by the ':' delimiter
            key, value = line.strip().split(';')
            
            # Remove leading/trailing spaces from the key and value
            key = key.strip()
            value = value.strip()
            
            # Convert the value to an appropriate data type if needed
            if value.startswith('[') and value.endswith(']'):
                # If the value is a list, convert it to a Python list
                value = eval(value)
            elif value.isdigit():
                # If the value is a number represented as a string, convert it to an integer
                value = int(value)
            elif '.' in value and all(part.isdigit() for part in value.split('.', 1)):
                # If the value is a float represented as a string, convert it to a float
                value = float(value)
            
            # Store the key-value pair in the dictionary
            data[key] = value

    # Print the dictionary
    return data








