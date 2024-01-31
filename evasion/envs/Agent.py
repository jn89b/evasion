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
        
        self.x = new_x
        self.y = new_y
        self.psi = new_psi
        
        #wrap angle between -pi and pi
        if self.psi > np.pi:
            self.psi = self.psi - 2*np.pi
        elif self.psi < -np.pi:
            self.psi = self.psi + 2*np.pi
        
        self.current_state = np.array([self.x, self.y, self.psi])

        
        return self.current_state
        
    def get_state(self) -> np.ndarray:
        """returns the current state of the agent"""
        return self.current_state
    
    
    def reset(self, set_random:bool=False, new_start:np.ndarray=None) -> None:
        """reset the agent to the start state"""
        if set_random and new_start is not None:
            self.start_state = new_start
            self.x = new_start[0]
            self.y = new_start[1]
            self.psi = new_start[2]
            self.current_state = new_start
        else:
            self.x = self.start_state[0]
            self.y = self.start_state[1]
            self.psi = self.start_state[2]
            self.current_state = self.start_state
        return self.current_state
