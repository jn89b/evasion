import numpy as np


class Agent:
    def __init__(self, init_states:np.ndarray, agent_params:dict) -> None:
        self.x = init_states[0]
        self.y = init_states[1]
        self.psi = init_states[2]
        self.current_state = init_states
        self.start_state = init_states
        self.agent_params = agent_params
        
        self.v_min = agent_params["v_min"]
        self.v_max = agent_params["v_max"]
        
        self.w_min = agent_params["w_min"]
        self.w_max = agent_params["w_max"]
        
        self.goal_x = agent_params["goal_x"]
        self.goal_y = agent_params["goal_y"]
        
        self.min_x = agent_params["min_x"]
        self.max_x = agent_params["max_x"]
        
        self.max_y = agent_params["max_y"]
        self.min_y = agent_params["min_y"]
        
        self.min_psi = agent_params["min_psi"]
        self.max_psi = agent_params["max_psi"]
        
        
    def move(self, action: np.ndarray) -> None:
        """Move the agent according to the action"""
        new_x = self.x + (action[0]*np.cos(self.psi))
        new_y = self.y + (action[0]*np.sin(self.psi))
        new_psi = self.psi + action[1]
        #new_psi = action[1]
        self.current_state = np.array([new_x, new_y, new_psi])
        
        self.x = new_x
        self.y = new_y
        self.psi = new_psi
        
        return self.current_state
        
    def get_state(self) -> np.ndarray:
        """returns the current state of the agent"""
        return self.current_state
    
    
    def reset(self) -> None:
        """reset the agent to the start state"""
        self.x = self.start_state[0]
        self.y = self.start_state[1]
        self.psi = self.start_state[2]
        self.current_state = self.start_state
        return self.current_state
