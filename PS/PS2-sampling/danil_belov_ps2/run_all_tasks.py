import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

from environment import State, ManipulatorEnv
from rrt import RRTPlanner
from angle_util import angle_linspace, angle_difference
from video_util import animate_plan


def l1_distance(state1, state2):
    return np.sum(np.abs(angle_difference(state2.angles, state1.angles)))


def weighted_distance(weights: np.ndarray):
    def dist_fn(state1, state2):
        diffs = np.abs(angle_difference(state2.angles, state1.angles))
        return np.sum(weights * diffs)
    return dist_fn


def check_collision_between_configs(env, state1, state2, n_steps = 50):
    angle_sequence = angle_linspace(state1.angles, state2.angles, n_steps)
    state_sequence = [State(angles) for angles in angle_sequence]
    
    for state in state_sequence:
        if env.check_collision(state):
            return True, state_sequence
    
    return False, state_sequence


def task_1a():
    
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)
    
    start_state = State(np.array(data["start_state"]))
    goal_state = State(np.array(data["goal_state"]))
    env = ManipulatorEnv(obstacles=np.array(data["obstacles"]),
                         initial_state=start_state,
                         collision_threshold=data["collision_threshold"])
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    env.state = start_state
    env.render(plt_show=False)
    plt.title("Start State", fontsize=14, fontweight='bold')
    
    plt.subplot(1, 2, 2)
    env.state = goal_state
    env.render(plt_show=False)
    plt.title("Goal State", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("task_1a_start_goal.png", dpi=150)
    print("Saved: task_1a_start_goal.png")
    plt.close()
    

def task_1b():
    
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)
    
    env = ManipulatorEnv(obstacles=np.array(data["obstacles"]),
                         initial_state=State(np.array([0, 0, 0, 0])),
                         collision_threshold=data["collision_threshold"])
    
    random.seed(42)
    np.random.seed(42)
    configs = []
    collision_status = []
    
    for i in range(4):
        angles = np.random.uniform(-180, 180, 4)
        state = State(angles)
        is_colliding = env.check_collision(state)
        configs.append(state)
        collision_status.append(is_colliding)

    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (state, is_colliding) in enumerate(zip(configs, collision_status)):
        env.state = state
        ax = axes[i]
        plt.sca(ax)
        env.render(plt_show=False)
        title = f"Config {i+1}: {'COLLIDING' if is_colliding else 'FREE'}"
        ax.set_title(title, fontsize=12, fontweight='bold',
                    color='red' if is_colliding else 'green')
    
    plt.tight_layout()
    plt.savefig("task_1b_random_configs.png", dpi=150)
    print("\nSaved: task_1b_random_configs.png")
    plt.close()


def task_2a():
    
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)
    
    env = ManipulatorEnv(obstacles=np.array(data["obstacles"]),
                         initial_state=State(np.array([0, 0, 0, 0])),
                         collision_threshold=data["collision_threshold"])
    
    np.random.seed(42)
    q1 = State(np.array([30.0, -20.0, 40.0, -30.0]))
    q2 = State(q1.angles + np.array([10.0, 5.0, -10.0, 8.0]))
    
    has_collision1, sequence1 = check_collision_between_configs(env, q1, q2, n_steps=50)
    
    q3 = State(np.array([90.0, -90.0, 90.0, -90.0]))
    q4 = State(np.array([-90.0, 90.0, -90.0, 90.0]))
    has_collision2, sequence2 = check_collision_between_configs(env, q3, q4, n_steps=50)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i, state in enumerate(sequence1[::5]):
        env.state = state
        env.render(plt_show=False)
    plt.title(f"No Collision Path\n(50 interpolation steps)", fontsize=12)
    
    plt.subplot(1, 2, 2)
    for i, state in enumerate(sequence2[::5]):
        env.state = state
        env.render(plt_show=False)
    plt.title(f"Collision Path\n(50 interpolation steps)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("task_2a_collision_check.png", dpi=150)
    print("\nSaved: task_2a_collision_check.png")
    plt.close()


def task_2b():

    
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)
    
    start_state = State(np.array(data["start_state"]))
    goal_state = State(np.array(data["goal_state"]))
    env = ManipulatorEnv(obstacles=np.array(data["obstacles"]),
                         initial_state=start_state,
                         collision_threshold=data["collision_threshold"])
    

    
    planner = RRTPlanner(env, distance_fn=l1_distance, max_angle_step=10.0)
    plan = planner.plan(start_state, goal_state, max_iterations=10000, goal_bias=0.1)
    

    animate_plan(env, plan, video_output_file="solve_4R.mp4")
    print("\nSaved: solve_4R.mp4")
    
    return planner, plan


def task_2c(planner, plan):
    print("\n=== Task 2C: Statistics ===")
    
    tree_size = planner.get_tree_size()
    path_states = len(plan)
    
    path_length = 0.0
    for i in range(len(plan) - 1):
        path_length += l1_distance(plan[i], plan[i+1])
    
    print(f"States visited (tree size): {tree_size} nodes")
    print(f"Final trajectory size: {path_states} states")
    print(f"Path length (L1 distance): {path_length:.2f} degrees")
    

def task_2d():
    
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)
    
    start_state = State(np.array(data["start_state"]))
    goal_state = State(np.array(data["goal_state"]))
    env = ManipulatorEnv(obstacles=np.array(data["obstacles"]),
                         initial_state=start_state,
                         collision_threshold=data["collision_threshold"])
    
    weight_configs = [
        (np.ones(4), "Uniform weights [1, 1, 1, 1]"),
        (np.array([2.0, 1.0, 1.0, 1.0]), "Emphasize joint 1 [2, 1, 1, 1]"),
        (np.array([1.0, 1.0, 2.0, 2.0]), "Emphasize joints 3-4 [1, 1, 2, 2]"),
        (np.array([0.5, 0.5, 1.5, 1.5]), "De-emphasize joints 1-2 [0.5, 0.5, 1.5, 1.5]"),
    ]
    
    results = []
    
    for weights, description in weight_configs:
        print(f"\nTesting: {description}")
        dist_fn = weighted_distance(weights)
        planner = RRTPlanner(env, distance_fn=dist_fn, max_angle_step=10.0)
        plan = planner.plan(start_state, goal_state, max_iterations=10000, goal_bias=0.1)
        
        tree_size = planner.get_tree_size()
        path_length = sum(l1_distance(plan[i], plan[i+1]) for i in range(len(plan)-1))
        
        results.append({
            'weights': weights,
            'description': description,
            'tree_size': tree_size,
            'path_length': path_length,
            'path_states': len(plan)
        })
        

    

def task_2e():
    
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)
    
    start_state = State(np.array(data["start_state"]))
    goal_state = State(np.array(data["goal_state"]))
    env = ManipulatorEnv(obstacles=np.array(data["obstacles"]),
                         initial_state=start_state,
                         collision_threshold=data["collision_threshold"])
    
    step_sizes = [5.0, 10.0, 15.0, 20.0]
    results = []
    
    for step_size in step_sizes:
        print(f"\nTesting step size: {step_size} degrees")
        planner = RRTPlanner(env, distance_fn=l1_distance, max_angle_step=step_size)
        plan = planner.plan(start_state, goal_state, max_iterations=10000, goal_bias=0.1)
        
        tree_size = planner.get_tree_size()
        path_length = sum(l1_distance(plan[i], plan[i+1]) for i in range(len(plan)-1))
        
        results.append({
            'step_size': step_size,
            'tree_size': tree_size,
            'path_length': path_length,
            'path_states': len(plan)
        })
        


def main():    
    task_1a()
    task_1b()
    
    task_2a()
    planner, plan = task_2b()
    task_2c(planner, plan)
    task_2d()
    task_2e()
    
    print("completed")


if __name__ == '__main__':
    main()

