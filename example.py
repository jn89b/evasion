# https://www.youtube.com/watch?v=_SWnNhM5w-g
# https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
# https://github.com/johnnycode8/gym_solutions/blob/main/mountain_car_q.py

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def compute_Q(alpha:float, current_q:float, reward:float, gamma:float, max_future_q:float):
    """
    compute the new Q value for a state-action pair 
    alpha: learning rate, alpha at 0 means no learning, 
        alpha at 1 means only care about new information
    current_q: current Q value
    reward: reward for taking the action
    gamma: discount factor, weight of future rewards, gamma at 0 means only care about immediate rewards, 
              gamma at 1 means care about future rewards as much as current rewards 
    max_future_q: maximum Q value for the next state
    """
    return (1-alpha)*current_q + alpha*(reward + gamma*max_future_q)

env = gym.make("MountainCar-v0", render_mode='human')

#go from continuous to discrete state space
low_states = env.observation_space.low
high_states = env.observation_space.high

NUM_DISCRETE_STATES = 20
NUM_EPISODES = 2500

TRAINING = True
pos_space = np.linspace(low_states[0], high_states[0], NUM_DISCRETE_STATES)
vel_space = np.linspace(low_states[1], high_states[1], NUM_DISCRETE_STATES)

if TRAINING:
    q_table = np.random.uniform(
        low=-2, high=0, size=(NUM_DISCRETE_STATES, NUM_DISCRETE_STATES, 
                              env.action_space.n))
else:
    q_table = np.load("q_table.npy")

learning_rate_alpha = 0.1
discount_factor_gamma = 0.9

epsilon = 1 # exploration vs exploitation
epsilon_decay_rate = 2/NUM_EPISODES
rng = np.random.default_rng()

#initialize rewards to empty list
rewards_per_episode = np.zeros(NUM_EPISODES)


for i in range(NUM_EPISODES):
    state = env.reset()[0]
    
    #put the state into discrete buckets
    state_p = np.digitize(state[0], pos_space)
    state_v = np.digitize(state[1], vel_space)
    
    terminated = False
    
    print(f"Episode {i}")
    
    rewards = 0
    
    while(not terminated and rewards>-1000):
        
        if TRAINING and rng.random() < epsilon:
            #explore choose a random action
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_p, state_v, :])
            
        new_state,reward,terminated,_,_ = env.step(action)
        new_state_p = np.digitize(new_state[0], pos_space)
        new_state_v = np.digitize(new_state[1], vel_space)

        if TRAINING:
            q_table[state_p, state_v, action] = compute_Q(
                learning_rate_alpha, q_table[state_p, state_v, action], 
                reward, discount_factor_gamma, 
                np.max(q_table[new_state_p, new_state_v, :]))
        
        state = new_state
        state_p = new_state_p
        state_v = new_state_v

        rewards+=reward

    epsilon = max(epsilon - epsilon_decay_rate, 0)

    rewards_per_episode[i] = rewards

env.close()

# Save Q table to file
if TRAINING:
    f = open('mountain_car.pkl','wb')
    pickle.dump(q_table, f)
    f.close()

mean_rewards = np.zeros(NUM_EPISODES)
for t in range(NUM_EPISODES):
    mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(mean_rewards)
plt.savefig(f'mountain_car.png')