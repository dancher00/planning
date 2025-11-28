import numpy as np
import pickle
from environment import State, ManipulatorEnv
from rrt import RRTPlanner
from video_util import animate_plan
from angle_util import angle_difference


# You are free to change any interfaces for your needs.


def l1_distance(state1, state2):
    return np.sum(np.abs(angle_difference(state2.angles, state1.angles)))


def weighted_distance(weights: np.ndarray = None):
    if weights is None:
        weights = np.ones(4)
    
    def dist_fn(state1, state2):
        diffs = np.abs(angle_difference(state2.angles, state1.angles))
        return np.sum(weights * diffs)
    
    return dist_fn


def main():
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)

    start_state = State(np.array(data["start_state"]))
    goal_state = State(np.array(data["goal_state"]))
    env = ManipulatorEnv(obstacles=np.array(data["obstacles"]),
                         initial_state=start_state,
                         collision_threshold=data["collision_threshold"])

    planner = RRTPlanner(env, distance_fn=l1_distance, max_angle_step=10.0)

    plan = planner.plan(start_state, goal_state)
    print(f"Tree size: {planner.get_tree_size()} nodes")
    print(f"Path length: {len(plan)} states")

    print("Generating video...")
    animate_plan(env, plan, video_output_file="solve_4R.mp4")
    print("Video saved: solve_4R.mp4")


if __name__ == '__main__':
    main()


