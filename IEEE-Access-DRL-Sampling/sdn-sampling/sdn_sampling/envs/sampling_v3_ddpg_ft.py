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
                mean_reward = np.mean(y[-100:])
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
            if (len(y) == 5e4):
                # plot
                cum_sum, mov_avgs = [0], []
                mov_N = 100
                for i, z in enumerate(y, 1):
                    cum_sum.append(cum_sum[i - 1] + z)
                    if i >= mov_N:
                        mov_avg = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                        mov_avgs.append(mov_avg)
                plt.xlim(1, 5e4)
                plt.xlabel("Steps")
                plt.ylabel("Reward")
                plt.plot(mov_avgs, 'bo-', label='DDPG')
                plt.legend(loc='lower right')
                plt.show()
                with open('./tmp/results/ddpg_rewards_v3.txt', 'w') as file:
                    file.write('\n'.join(map(str, mov_avgs)))

        return True

# Monitor training using DDPG
# Create log dir
log_dir = "tmp/log/sampling_v3"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('sdnsampling-v3')
env = Monitor(env, log_dir)

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=0)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Train the agent
time_steps = 5e4
model.learn(total_timesteps=int(time_steps), callback=callback)
model.save("ddpg_sampling_mal_ta3_ft_v3")

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG Sampling")
plt.show()

# Test by trained model
# env = gym.make('sdnsampling-v3')
# model = DDPG.load("ddpg_sampling_mal_ta3_ft_v1")
# plot_result = []
# n_steps = 50
# obs = env.reset()
# for step in range(n_steps):
#     action, _states = model.predict(obs)
#     print('------------------------------------------------------')
#     print("Step {}".format(step + 1))
#     obs, reward, done, info = env.step(action)
#     print('obs=', obs, 'reward=', reward, 'done=', done, 'info=', info)
#     # env.render()

# DDPG_1
# env = gym.make('sdnsampling-v3')
# n_actions = env.action_space.shape[-1]
# # Add some param noise for exploration
# param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
#
# model = DDPG(LnMlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=None)
# model.learn(total_timesteps=30000)
# # model.save("ddpg_sampling")
# # del model  # remove to demonstrate saving and loading

# Test by trained model
# env = gym.make('sdnsampling-v3')
# model = DDPG.load("ddpg_sampling_mal_ta3_ft_v1")
# plot_result = []
# n_steps = 500
# obs = env.reset()
# for step in range(n_steps):
#     action, _states = model.predict(obs)
#     print('------------------------------------------------------')
#     print("Step {}".format(step + 1))
#     obs, reward, done, info = env.step(action)
#     print('obs=', obs, 'reward=', reward, 'done=', done)
#     # env.render()

#     plot_result.append(reward)
    # avg_reward = total_reward / steps
    # if done:
    #     print("Goal reached!", "reward=", reward)
    #     break
#
# # plot and save rewards
# cum_sum, mov_avgs = [0], []
# mov_N = 100
# for i, x in enumerate(plot_result, 1):
#     cum_sum.append(cum_sum[i-1] + x)
#     if i >= mov_N:
#         mov_avg = (cum_sum[i] - cum_sum[i-mov_N])/mov_N
#         mov_avgs.append(mov_avg)
# plt.xlim(1, n_steps)
# plt.xlabel("Steps")
# plt.ylabel("Reward")
# plt.plot(mov_avgs, 'bo-', label = 'DDPG')
# plt.legend(loc='lower right')
# plt.show()
# with open('./tmp/results/DDPG_rewards_1.txt', 'w') as file:
#     file.write('\n'.join(map(str, mov_avgs)))