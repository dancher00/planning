from typing import List, Callable, Tuple, Optional
import numpy as np
from angle_util import angle_linspace, angle_difference
from environment import State, ManipulatorEnv


class RRTPlanner:

    def __init__(self,
                 env: ManipulatorEnv,
                 distance_fn: Callable,
                 max_angle_step: float = 10.0):
        """
        :param env: manipulator environment
        :param distance_fn: function distance_fn(state1, state2) -> float
        :param max_angle_step: max allowed step for each joint in degrees
        """
        self._env = env
        self._distance_fn = distance_fn
        self._max_angle_step = max_angle_step


        self._nodes = []
        self._parents = []
        self._n_steps_collision_check = 50

    def _check_collision_between_configs(self, state1, state2):
        angle_sequence = angle_linspace(state1.angles, state2.angles, self._n_steps_collision_check)
        state_sequence = [State(angles) for angles in angle_sequence]
        
        for state in state_sequence:
            if self._env.check_collision(state):
                return True
        
        return False

    def _nearest_node(self, target_state) -> int:
        if len(self._nodes) == 0:
            return -1
        
        distances = [self._distance_fn(node, target_state) for node in self._nodes]
        return np.argmin(distances)

    def _steer(self, from_state, to_state) -> State:
        angle_diffs = angle_difference(to_state.angles, from_state.angles)
        
        max_diffs = np.clip(angle_diffs, -self._max_angle_step, self._max_angle_step)
        new_angles = from_state.angles + max_diffs
        
        new_angles = ((new_angles + 180) % 360) - 180
        
        return State(new_angles)

    def _is_goal_reached(self, state, goal_state, threshold = 5.0):
        return self._distance_fn(state, goal_state) < threshold

    def plan(self,
             start_state,
             goal_state,
             max_iterations = 10000,
             goal_bias = 0.1) -> List[State]:
        """
        RRT algorithm implementation. """

        self._nodes = [start_state]
        self._parents = [-1]
        
        for iteration in range(max_iterations):
            if iteration % 1000 == 0:
                print(f"RRT iteration: {iteration}/{max_iterations}, tree size: {len(self._nodes)}")
            
            if np.random.random() < goal_bias:
                q_rand = goal_state
            else:
                random_angles = np.random.uniform(-180, 180, 4)
                q_rand = State(random_angles)
            
            nearest_idx = self._nearest_node(q_rand)
            q_near = self._nodes[nearest_idx]
            
            q_new = self._steer(q_near, q_rand)
            
            if not self._check_collision_between_configs(q_near, q_new):
                self._nodes.append(q_new)
                self._parents.append(nearest_idx)
                
                if self._is_goal_reached(q_new, goal_state):
                    if not self._check_collision_between_configs(q_new, goal_state):
                        self._nodes.append(goal_state)
                        self._parents.append(len(self._nodes) - 2)
                        
                        path = self._reconstruct_path(len(self._nodes) - 1)
                        print(f"Goal reached at iteration {iteration}!")
                        return path
        
        print(f"Warning: Max iterations ({max_iterations}) reached. Returning path to closest node.")
        closest_to_goal_idx = self._nearest_node(goal_state)
        path = self._reconstruct_path(closest_to_goal_idx)
        return path

    def _reconstruct_path(self, goal_idx):
        path = []
        current_idx = goal_idx
        
        while current_idx != -1:
            path.append(self._nodes[current_idx])
            current_idx = self._parents[current_idx]
        
        path.reverse()
        return path

    def get_tree_size(self):
        return len(self._nodes)
