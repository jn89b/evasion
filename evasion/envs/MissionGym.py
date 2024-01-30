import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt

from typing import Any, SupportsFloat
from gymnasium import spaces # https://www.gymlibrary.dev/api/spaces/
from evasion.envs.graphics import GameDrawer
from evasion.envs.Agent import Agent


# https://github.com/asack20/RL-in-Pursuit-Evasion-Game/blob/master/src/robot.py

# https://www.gymlibrary.dev/api/core/


"""
States will be x,y,theta and mapped  to a continous space

Actions will consist of linear and angular velocity:
    - [v_min , v_max] and [w_min, w_max] in discrete steps
"""


#this will be abstracted to an Agent class

class MissionGym(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 7}

    def __init__(self, evader:Agent, num_discrete_actions:int=10, 
                 render_mode:str=None, render_fps:int=7):
        super(MissionGym, self).__init__()
            
        # observation space of the evader
        self.v_range = [evader.v_min, evader.v_max] #velocity range
        self.w_range = [evader.w_min, evader.w_max] #angular velocity range
        self.v_space = np.linspace(evader.v_min, evader.v_max, num_discrete_actions)
        self.w_space = np.linspace(evader.w_min, evader.w_max, num_discrete_actions)
        
        self.evader = evader

        # action space of the evader
        self.total_actions = num_discrete_actions**2
        self.action_space = spaces.Discrete(self.total_actions)
        
        #map action space to combination of linear and angular velocity
        self.action_map = {}
        for i in range(num_discrete_actions):
            for j in range(num_discrete_actions):
                self.action_map[i*num_discrete_actions + j] = [self.v_space[i], self.w_space[j]]
        
    
        self.goal_location = np.array([evader.goal_x, evader.goal_y])
    
        # observation space
        self.evader_observation_space = spaces.Box(low=np.array([evader.min_x, evader.min_y, evader.min_psi]), 
                                                   high=np.array([evader.max_x, evader.max_y, evader.max_psi]), 
                                                   dtype=np.float32)
        
        self.observation_space = spaces.Dict(
            {
            "evader": self.evader_observation_space,
            }
        )
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.time_limit = 1000
        self.game_window = None 
        self.clock = None
        self.buffer = 200
        self.game_renderer = GameDrawer() 
        self.width = 1000 + self.buffer  #int(abs(evader.max_x - evader.min_x))
        self.height = 1000 + self.buffer#int(abs(evader.max_y - evader.min_y))
        self.render_fps = render_fps
        
        self.old_distance = self.compute_distance_cost(self.evader.get_state())
        self.total_reward = 0
        
    def print_info(self):
        print("observation space: ", self.observation_space)
        print("action space: ", self.action_space)
        print("evader observation space: ", self.observation_space["evader"])
        print("evader action space: ", self.action_map)
        print("goal location: ", self.goal_location)
        print("evader: ", self.evader)
        
    
    def discretize_action(self, action: int):
        """returns the discrete action from the continuous action"""
        #check if the action is within the range
        
        vel, ang_vel = self.action_map[int(action)]        
        return vel, ang_vel
    
    def __get_observation(self) -> dict:
        return {"evader": self.evader.get_state()}
    
    def __get_info(self) -> dict[str, Any]:
        agent_location = self.evader.get_state()
        distance = np.linalg.norm(agent_location[:2] - self.goal_location)
        info_dict = {"distance": distance}
        return info_dict
    
    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        
        self.time_limit -= 1

        
        velocity, angular_velocity = self.discretize_action(action)
        self.state = self.evader.move(np.array([velocity, angular_velocity]))
        
        #make sure the state is within the bounds 
        self.state[0] = np.clip(self.state[0], self.evader.min_x, self.evader.max_x)
        self.state[1] = np.clip(self.state[1], self.evader.min_y, self.evader.max_y)
        
        #wrap the angle between min and max
        if self.state[2] > self.evader.max_psi:
            self.state[2] = self.evader.min_psi + (self.state[2] - self.evader.max_psi)
        elif self.state[2] < self.evader.min_psi:
            self.state[2] = self.evader.max_psi - (self.evader.min_psi - self.state[2])
            
        distance = self.compute_distance_cost(self.state)
        delta_distance = distance - self.old_distance
        #if the distance is decreasing, reward the agent
        if delta_distance > 0:
            #sign of the distance is negative, so we want to reward the agent
            #reward = - delta_distance
            reward = 1
        else:
            #reward = delta_distance
            reward = -1
        
        reward = 1/distance
        # reward += -0.1

        #set the reward based on the distance, we want to minimize the distance
        done = False

        if distance < 10:
            print("goal reached", distance, self.state)
            done = True
            reward += 100
            print("done", done)
            self.reset()
            
        # if self.time_limit <= 0:
        #     done = True
        #     reward += -10
        #     self.reset()
        
        info = self.__get_info()
        observation =  self.__get_observation()
        
        self.old_distance = distance
        
        return observation, reward, done, False, info

    def compute_distance_cost(self, state: np.ndarray) -> float:
        dx = state[0] - self.goal_location[0]
        dy = state[1] - self.goal_location[1]
        distance = np.sqrt(dx**2 + dy**2)
        return distance

    def reset(self, seed=None) -> Any:
        #reset the evader to the start state
        self.evader.reset()
        observation = self.__get_observation()
        info = self.__get_info()
        
        if self.render_mode == "human":
            self.__render_frame()

        return observation, info
        
    
    def render(self, mode: str = 'pass') -> None:
        if self.render_mode == "human":
            return self.__render_frame()
        
    def __render_frame(self):
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py 
        if self.game_window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.game_window = pygame.display.set_mode((self.width, self.height))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # do your drawing here
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((255, 255, 255))
        
        [end, right_side, left_side] = self.game_renderer.draw_arrow(self.evader.current_state[:2], 
                                                                     np.rad2deg(self.evader.current_state[2]))
        
        # print("evader state: ", self.evader.current_state[:2])
        pygame.draw.line(canvas, (0, 0, 255), self.evader.current_state[:2], end, 5)
        pygame.draw.polygon(canvas, (0, 0, 255), [end, right_side, left_side])
        
        #draw the goal location
        pygame.draw.circle(canvas, (255, 0, 0), self.goal_location.astype(int), 10)
        
        #write the position of the evader
        # font = pygame.font.Font('freesansbold.ttf', 32)
        # text = font.render(f'x: {self.evader.current_state[0]:.2f}, y: {self.evader.current_state[1]:.2f}', True, (0, 0, 0))
        # textRect = text.get_rect()
        # # put in lower left corner
        # textRect.bottomleft = (10, self.height - 10)
        # canvas.blit(text, textRect)
        
        if self.render_mode == "human":
            # pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            self.game_window.blit(canvas, canvas.get_rect())

            # pygame.display.flip()
            # # The following line copies our drawings from `canvas` to the visible window
            pygame.event.pump()
            pygame.display.update()

            # # We need to ensure that human-rendering occurs at the predefined framerate.
            # # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            
    
    def close(self) -> None:
        if self.game_window is not None:
            pygame.display.quit()
            pygame.quit()
            self.game_window = None
            self.clock = None
        
        
