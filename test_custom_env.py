import numpy as np
import pygame
import gymnasium as gym
from evasion.envs.Agent import Agent
from evasion.envs.MissionGym import MissionGym
#from config import v_min, v_max, w_min, w_max, min_x, min_y, min_psi, max_x, max_y, max_psi, goal_x, goal_y, total_actions
# from custom_env.custom_env import MissionGym
# from custom_env.custom_env import Agent

start_x = 500
start_y = 500
start_psi = np.deg2rad(0)

v_min = 0.0
v_max = 10.0
w_min = np.deg2rad(-45)
w_max = np.deg2rad(45)
goal_x = 800
goal_y = 800
min_x = 0
max_x = 1000
min_y = 0
max_y = 1000
min_psi = np.deg2rad(-180)
max_psi = np.deg2rad(180)

agent_params = {
    "v_min": v_min,
    "v_max": v_max,
    "w_min": w_min,
    "w_max": w_max,
    "goal_x": goal_x,
    "goal_y": goal_y,
    "min_x": min_x,
    "max_x": max_x,
    "max_y": max_y,
    "min_y": min_y,
    "min_psi": min_psi,
    "max_psi": max_psi,
}

init_states = np.array([start_x, start_y, start_psi])
evader = Agent(init_states, agent_params)
print("evader: ", evader.agent_params)
# env = MissionGym(evader, render_mode="human")
env = gym.make('evasion/MissionGym-v0', render_mode='human', 
               evader=evader, render_fps=15, use_random_start=True)
# env.print_info()
action_space_index = env.action_space.sample()
print("random sample space", env.action_space.sample())
# control = env.action_map[action_space_index]
print("render mode: ", env.render_mode)
env.reset()

done = False
old_state = evader.get_state()
reset_modulus = 100
action_space_index = np.array([-1, 0])
for i in range(500):
# while True:
    
    # print("action space index: ", action_space_index)
    # control = env.action_map[action_space_index]
    #print("control: ", control)
    new_state, reward, done, _, info = env.step(action_space_index)
    # print("reward: ", reward)
    #print("new state: ", new_state)
    old_state = new_state
    env.render()
    
    # if i % reset_modulus == 0:
    #     random_number = np.random.randint(0, 100)
    #     action_space_index = np.array([-1, 1])
    #     env.reset(seed=random_number)
    #     print("reset")
        
