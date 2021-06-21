import os
import sdn_sampling
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-10:])
                #print(mean_reward)
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

            # Plot and save rewards
            if (len(y) == 1e4):
                # plot
                cum_sum, mov_avgs = [0], []
                mov_N = 100
                for i, z in enumerate(y, 1):
                    cum_sum.append(cum_sum[i - 1] + z)
                    if i >= mov_N:
                        mov_avg = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                        mov_avgs.append(mov_avg)
                plt.xlim(1, 1e4)
                plt.xlabel("Steps")
                plt.ylabel("Reward")
                plt.plot(mov_avgs, 'bo-', label='DDPG')
                plt.legend(loc='lower right')
                plt.show()
                with open('./tmp/results/sampling_MTD_exp_v0_ddpg_ft_rewards.txt', 'w') as file:
                    file.write('\n'.join(map(str, mov_avgs)))

        return True

# env = gym.make('sdnsampling-MTD-v0')
# obs = env.reset
# print('initial obs', obs)
#
# plot_result = []
# n_steps = 50
# action = env.action_space.sample()
#
# for step in range(n_steps):
#   action = env.action_space.sample()
#   print('------------------------------------------------------')
#   print('action=', action)
#   print("Step {}".format(step + 1))
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   # env.render()
#   plot_result.append(reward)
#   # if done:
#   #   print("Goal reached!", "reward=", reward)
#   #   break
#
# # plot and save rewards
# cum_sum, mov_avgs = [0], []
# mov_N = 10
# for i, x in enumerate(plot_result, 1):
#     cum_sum.append(cum_sum[i-1] + x)
#     if i >= mov_N:
#         mov_avg = (cum_sum[i] - cum_sum[i-mov_N])/mov_N
#         mov_avgs.append(mov_avg)
# plt.xlim(1, n_steps)
# plt.xlabel("Steps")
# plt.ylabel("Reward")
# plt.plot(mov_avgs, 'bo-', label = 'Random')
# plt.legend(loc='lower right')
# plt.show()
# with open('./tmp/results/sampling_MTD_v0_ddpg_ft_rewards.txt', 'w') as file:
#     file.write('\n'.join(map(str, mov_avgs)))
#

# Monitor training using DDPG
# Create log dir
log_dir = "tmp/log/sampling_MTD_exp_v0"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('sdnsampling-MTD-exp-v0')
env = Monitor(env, log_dir)

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlSpPolicy with layer normalization
model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=0)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Train the agent
time_steps = 1e4
model.learn(total_timesteps=int(time_steps), callback=callback)
model.save("sampling_MTD_exp_v0_ddpg_ft_1e3")

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG Sampling MTD")
plt.show()
