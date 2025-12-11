import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *
#import pickle
from mdp import MDP
from vi import VI


data = np.load('data_ps3.npz')
environment_grid = data['environment']


# (row index, colum index). In the image row corresponds to y, and colum to s.
s_ini = (0,0)
goal = (19,17)
epsilon = 0.4 #Propagation probability (see utils)

environment = Environment(environment_grid,s_ini,goal,epsilon)



# Visualization
# ======================================
if 0:
    im = environment.plot_enviroment(s_ini, goal)
    plt.matshow(im)
    plt.show()


# task 1 VI, Gopt
# ======================================
print("Running Value Iteration...")
vi = VI(environment, goal)
G_opt = vi.calculate_value_function()
vi.calculate_policy()

# Visualize G*
plt.figure(figsize=(10, 8))
plt.imshow(G_opt, cmap='viridis')
plt.colorbar(label='Cost-to-go G*')
plt.title('Value Iteration: Optimal Cost-to-go G*')
plt.savefig('vi_G_opt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved vi_G_opt.png")

# task 2 MDP
# ======================================
print("Running MDP...")
mdp = MDP(environment, goal, gamma=0.99)
v_opt = mdp.calculate_value_function()
mdp.calculate_policy()

# Visualize v*
plt.figure(figsize=(10, 8))
plt.imshow(v_opt, cmap='viridis')
plt.colorbar(label='Value function v*')
plt.title('MDP: Optimal Value Function v*')
plt.savefig('mdp_v_opt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved mdp_v_opt.png")

# Plan visualization for VI
# ======================================
print("Generating VI video...")
fig = plt.figure()
imgs = []
s = s_ini
environment.reset(s)
for iters in range(100):
    print('state ', s, ' iters ', iters)
    im = environment.plot_enviroment(s, goal)
    plot = plt.imshow(im, animated=True)
    imgs.append([plot])
    # Calculate plan based on VI policy
    a = vi.policy(s)
    # NOTE, ONLY for visualizations, the noise has been reduced, but it should be different than for calculating the MDP
    s, _, safe_propagation, success = environment.step(action_space[a], epsilon=0.001)
    if not safe_propagation:
        print('Collision!!', s)
        break
    print('iters', iters, ' action ', a, ' state ', s)
    if success:
        print('Goal achieved in ', iters)
        im = environment.plot_enviroment(s, goal)
        plot = plt.imshow(im, animated=True)
        imgs.append([plot])
        break

ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat=False)
ani.save('plan_vi.mp4', writer='ffmpeg')
plt.close()
print("Saved plan_vi.mp4")

# Plan visualization for MDP
# ======================================
print("Generating MDP video...")
fig = plt.figure()
imgs = []
s = s_ini
environment.reset(s)
for iters in range(100):
    print('state ', s, ' iters ', iters)
    im = environment.plot_enviroment(s, goal)
    plot = plt.imshow(im, animated=True)
    imgs.append([plot])
    # Calculate plan based on MDP policy
    a = mdp.policy(s)
    # NOTE, ONLY for visualizations, the noise has been reduced, but it should be different than for calculating the MDP
    s, _, safe_propagation, success = environment.step(action_space[a], epsilon=0.001)
    if not safe_propagation:
        print('Collision!!', s)
        break
    print('iters', iters, ' action ', a, ' state ', s)
    if success:
        print('Goal achieved in ', iters)
        im = environment.plot_enviroment(s, goal)
        plot = plt.imshow(im, animated=True)
        imgs.append([plot])
        break

ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat=False)
ani.save('plan_mdp.mp4', writer='ffmpeg')
plt.close()
print("Saved plan_mdp.mp4")

