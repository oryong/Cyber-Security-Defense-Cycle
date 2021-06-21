import os
import sdn_sampling
import gym
import numpy as np
import matplotlib.pyplot as plt

# Randomly select sampling points, sampling reduction rate, and TA selection
env = gym.make('sdnsampling-v0')
obs = env.reset
print('initial obs', obs)

plot_result = []
n_steps = 50000
action = env.action_space.sample()
for step in range(n_steps):
  action = env.action_space.sample()
  print('------------------------------------------------------')
  #print('action=', action)
  print("Step {}".format(step + 1))
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  # env.render()
  plot_result.append(reward)
  # if done:
  #   print("Goal reached!", "reward=", reward)
  #   break

# plot and save rewards
cum_sum, mov_avgs = [0], []
mov_N = 100
for i, x in enumerate(plot_result, 1):
    cum_sum.append(cum_sum[i-1] + x)
    if i >= mov_N:
        mov_avg = (cum_sum[i] - cum_sum[i-mov_N])/mov_N
        mov_avgs.append(mov_avg)
plt.xlim(1, n_steps)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.plot(mov_avgs, 'bo-', label = 'Random')
plt.legend(loc='lower right')
plt.show()
with open('./tmp/results/random_rewards_ft.txt', 'w') as file:
    file.write('\n'.join(map(str, mov_avgs)))