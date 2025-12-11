import numpy as np
from utils import *


class MDP:

    def __init__(self,
                 env: Environment,
                 goal: tuple,
                 gamma: float = 0.99):
        """
        env is the grid enviroment
        goal is the goal state
        gamma is the discount factor
        """
        self._env = env
        self._goal = goal
        self._gamma = gamma
        self._V = np.zeros(env.shape)
        self._policy = np.zeros(self._env.shape, 'b') #type byte (or numpy.int8)

    def calculate_value_function(self):
        """
        This function uses the Value Iteration algorithm to fill in the
        optimal value function
        
        Mathematical Formulation (Task 2.A):
        The optimal value function v*(s) satisfies the Bellman equation:
        v*(s) = max_{a in A} [r(s,a) + γ * Σ_{s'} P(s'|s,a) * v*(s')]
        where:
        - r(s,a) is the reward: 1 if s' = goal, -1 if collision/out of bounds, 0 otherwise
        - γ is the discount factor (gamma)
        - P(s'|s,a) is the probabilistic transition function
        - v*(goal) = 1 (terminal state with reward)
        """
        # Initialize: v*(goal) = 1, all other states = 0
        self._V = np.zeros(self._env.shape)
        self._V[self._goal] = 1.0
        
        # Value Iteration: iterate until convergence
        max_iterations = 100
        for iteration in range(max_iterations):
            V_new = np.copy(self._V)
            
            # Update all states
            for i in range(self._env.shape[0]):
                for j in range(self._env.shape[1]):
                    s = (i, j)
                    
                    # Goal state remains 1
                    if s == self._goal:
                        continue
                    
                    # Skip obstacles
                    if not self._env.state_consistency_check(s):
                        continue
                    
                    # Find maximum value over all actions
                    max_value = -np.inf
                    for a_idx, a in enumerate(action_space):
                        # Get probabilistic transitions
                        state_list, prob_list = self._env.probabilistic_transition_function(s, a)
                        
                        # Calculate expected value: r(s,a) + gamma * sum_s' P(s'|s,a) * v*(s')
                        expected_value = 0.0
                        for s_next, prob in zip(state_list, prob_list):
                            # Check if state is valid and within bounds
                            is_valid = self._env.state_consistency_check(s_next)
                            in_bounds = (0 <= s_next[0] < self._env.shape[0] and 
                                        0 <= s_next[1] < self._env.shape[1])
                            
                            if is_valid and in_bounds:
                                if s_next == self._goal:
                                    reward = 1.0
                                else:
                                    reward = 0.0
                                v_next = self._V[s_next]
                            else:
                                # Obstacle or out of bounds
                                reward = -1.0
                                v_next = 0.0  # Terminal state, no future value
                            
                            expected_value += prob * (reward + self._gamma * v_next)
                        
                        max_value = max(max_value, expected_value)
                    
                    V_new[s] = max_value
            
            # Check for convergence
            if np.allclose(self._V, V_new, atol=1e-6):
                print(f"MDP converged after {iteration + 1} iterations")
                break
            
            self._V = V_new
        
        return self._V

    def calculate_policy(self):
        """
        Only to be run AFTER Vopt has been calculated.
        
        output:
        policy: a map from each state s to the greedy best action a to execute
        
        Mathematical Formulation (Task 2.B):
        The greedy deterministic policy π*_MDP(s) is obtained from v* as:
        π*_MDP(s) = argmax_{a in A} [r(s,a) + γ * Σ_{s'} P(s'|s,a) * v*(s')]
        This selects the action that maximizes the expected immediate reward plus 
        the discounted expected future value.
        """
        # Calculate greedy policy from v*
        for i in range(self._env.shape[0]):
            for j in range(self._env.shape[1]):
                s = (i, j)
                
                # Skip obstacles
                if not self._env.state_consistency_check(s):
                    continue
                
                # Find action that maximizes value
                best_action_idx = 0
                max_value = -np.inf
                
                for a_idx, a in enumerate(action_space):
                    # Get probabilistic transitions
                    state_list, prob_list = self._env.probabilistic_transition_function(s, a)
                    
                    # Calculate expected value
                    expected_value = 0.0
                    for s_next, prob in zip(state_list, prob_list):
                        # Check if state is valid and within bounds
                        is_valid = self._env.state_consistency_check(s_next)
                        in_bounds = (0 <= s_next[0] < self._env.shape[0] and 
                                    0 <= s_next[1] < self._env.shape[1])
                        
                        if is_valid and in_bounds:
                            if s_next == self._goal:
                                reward = 1.0
                            else:
                                reward = 0.0
                            v_next = self._V[s_next]
                        else:
                            # Obstacle or out of bounds
                            reward = -1.0
                            v_next = 0.0  # Terminal state, no future value
                        
                        expected_value += prob * (reward + self._gamma * v_next)
                    
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action_idx = a_idx
                
                self._policy[s] = best_action_idx
        
        return self._policy

    def policy(self,state:tuple) -> int:
        """
        returns the action according to the policy
        """
        return self._policy[state]

