import numpy as np
from utils import *


class VI:
    def __init__(self,
                 env: Environment,
                 goal: tuple):
        """
        env is the grid enviroment, as defined in utils
        goal is the goal state
        """
        self._env = env
        self._goal = goal
        self._G = np.ones(self._env.shape)*1e2
        self._policy = np.zeros(self._env.shape, 'b') #type byte (or numpy.int8)


    def calculate_value_function(self):
        """
        env is the grid enviroment
        goal is the goal state
            
        output:
        G: Optimal cost-to-go
        
        Mathematical Formulation (Task 1.B):
        The optimal cost-to-go G*(s) satisfies the Bellman equation:
        G*(s) = min_{a in A} [l(s,a) + G*(f(s,a))]
        where:
        - l(s,a) = 1 if propagation is possible (no obstacle/out of bounds)
        - f(s,a) is the deterministic transition function
        - G*(goal) = 0 (terminal state)
        """
        # Initialize: G*(goal) = 0, all other states = large value
        self._G = np.ones(self._env.shape) * 1e6
        self._G[self._goal] = 0.0
        
        # Value Iteration: iterate until convergence
        max_iterations = 100
        for iteration in range(max_iterations):
            G_new = np.copy(self._G)
            
            # Update all states except goal
            for i in range(self._env.shape[0]):
                for j in range(self._env.shape[1]):
                    s = (i, j)
                    
                    # Goal state remains 0
                    if s == self._goal:
                        continue
                    
                    # Skip obstacles
                    if not self._env.state_consistency_check(s):
                        continue
                    
                    # Find minimum cost over all actions
                    min_cost = np.inf
                    for a_idx, a in enumerate(action_space):
                        s_next, success = self._env.transition_function(s, a)
                        
                        if success:
                            # Cost l(s,a) = 1 if propagation is possible
                            cost = 1.0 + self._G[s_next]
                        else:
                            # If propagation fails, cost is infinite (or very large)
                            cost = 1e6
                        
                        min_cost = min(min_cost, cost)
                    
                    G_new[s] = min_cost
            
            # Check for convergence
            if np.allclose(self._G, G_new, atol=1e-6):
                print(f"VI converged after {iteration + 1} iterations")
                break
            
            self._G = G_new
        
        return self._G
        
    def calculate_policy(self):
        """
        G: optimal cot-to-go function (needed to be calcualte in advance)
        
        output:
        policy: a map from each state x to the best action a to execcute
        
        Mathematical Formulation (Task 1.D):
        The optimal policy π*_VI(s) is obtained from G* as:
        π*_VI(s) = argmin_{a in A} [l(s,a) + G*(f(s,a))]
        This selects the action that minimizes the immediate cost plus the 
        optimal cost-to-go from the next state.
        """
        # Calculate optimal policy from G*
        for i in range(self._env.shape[0]):
            for j in range(self._env.shape[1]):
                s = (i, j)
                
                # Skip obstacles
                if not self._env.state_consistency_check(s):
                    continue
                
                # Find action that minimizes cost
                best_action_idx = 0
                min_cost = np.inf
                
                for a_idx, a in enumerate(action_space):
                    s_next, success = self._env.transition_function(s, a)
                    
                    if success:
                        cost = 1.0 + self._G[s_next]
                    else:
                        cost = 1e6
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_action_idx = a_idx
                
                self._policy[s] = best_action_idx
        
        return self._policy
        
    def policy(self,state:tuple) -> int:
        """
        returns the action according to the policy
        """
        return self._policy[state]

