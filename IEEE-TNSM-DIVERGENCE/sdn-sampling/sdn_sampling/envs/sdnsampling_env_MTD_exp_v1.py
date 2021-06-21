import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import random
from math import exp
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import subprocess

class SdnSamplingEnv_MTD_exp_v1(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        # set result var.
        self.attacked_groups = [0, 0, 0, 0]
        self.routing_candidate = 0
        self.malicious_rate = 0.01
        self.mal_samp = 0.0
        self.F_samp = 0.0
        self.reward_f = 0.0
        self.res_util_TAs = []
        self.reward_u = 0.0
        self.mean_overhead = 0.0
        self.reward_v = 0.0
        self.reward_total = 0.0

        # set plot var.
        self.result_F_samp = []
        self.result_mal_f = []
        self.result_f = []
        self.result_res_util_TAs = []
        self.result_u = []
        self.result_mean_overhead = []
        self.result_v = []
        self.result_total = []

        # set topology info.
        self.num_steps = 10000
        # self.topology_path = './tmp/topo/routing_matrix_30_50000.txt'
        # self.num_switches = 30
        # self.num_flows = 50000
        self.topology_path = './tmp/topo/routing_matrix_10_10000.txt'
        self.num_switches = 10
        self.num_flows = 10000
        self.num_TA = 4
        self.num_group = 4
        self.total_cap_TA = 100000000.0
        #self.total_cap_TA = 10000000.0
        self.r = 0.5

        self.min_flow_rate = 1.0
        self.max_flow_rate = 40.0
        self.min_sampled_flows = 0
        self.max_sampled_flows = self.num_flows

        self.min_hop = 1
        self.max_hop = 3

        self.routing_matrix, \
        self.traffic_matrix, \
        self.rate_flows, \
        self.priority_flows, \
        self.routed_num_flows, \
        self.hop_mat = self._set_topology()
        #self.mal_flows = self._set_mal_flows()

        # number of flows at switch
        tmp_rm = np.asmatrix(self.routing_matrix)
        self.num_flows_at_switch = np.squeeze(
            np.asarray(tmp_rm.sum(axis=0))
        )
        print('|F^tot|=', self.num_flows_at_switch)
        print('routed_num_flows=', self.routed_num_flows)

        # data rate at switch
        tmp_tm = np.asmatrix(self.traffic_matrix)
        self.sum_flows_rate_at_switch = np.squeeze(
            np.asarray(tmp_tm.sum(axis=0))
        )
        print('data_rate_o=', self.sum_flows_rate_at_switch)
        print('total_data_rate=', sum(self.sum_flows_rate_at_switch))
        print('hop matrix:', self.hop_mat)

        self.max_total_rate_g = sum(self.sum_flows_rate_at_switch)

        # Set capacity of traffic analyzers
        self.cap_TA = self.total_cap_TA / self.num_TA

        # Set state space s_t
        # self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), \
        #                             high=np.array([self.max_total_rate_g, self.max_total_rate_g, self.max_total_rate_g, self.max_total_rate_g,
        #                                            1, 1, 1, 1]), \
        #                             dtype=np.float)

        self.observation_space = spaces.Box(low=np.array([0.0]*self.num_group*2), \
                                            high=np.array(
                                                [self.max_total_rate_g, self.max_total_rate_g, self.max_total_rate_g,
                                                 self.max_total_rate_g,
                                                 1, 1, 1, 1]), \
                                            dtype=np.float)

        # Set action space a_t
        self.action_space = spaces.Box(low=np.array([0.0]*self.num_group), \
                                    high=np.array([1]*self.num_group), \
                                    dtype=np.float)

        # Set initial state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        # Set initial state
        self.state = self.sum_flows_rate_at_switch[-self.num_group:]
        self.state = np.append(self.state, self.action_space.sample())
        # print(self.state)
        done = False

        return np.array(self.state)

    def step(self, action):
        print('[Steps]', len(self.result_total) + 1)
        # Scale action
        # print('selected action:', action)

        # Get next state according to the current action
        self.state = self._compute_observation(action)
        # print('next state:', self.state)

        # Get reward with current state and action
        self.reward_total, self.mal_samp, self.reward_v = self._compute_reward(self.state, action)
        #print('reward at st at:', reward)

        # Set time-step per episode
        done = True
        #done = bool(self.reward_total >= 1)
        info = {}

        # save rewards
        self.result_total.append(self.reward_total)
        self.result_mal_f.append(self.mal_samp)
        self.result_v.append(self.reward_v)
        # print(self.result_f)
        # print(self.result_u)
        # print(self.result_v)
        # print(self.result_total)
        # print('-----------------------')
        # print('reward_total:', self.reward_total)
        print('-----------------------')
        print('mal_sampled', self.mal_samp)
        print('num_sampled:', self.F_samp)
        print('reward_v:', self.reward_v)
        print('reward_total:', self.reward_total)

        if len(self.result_total) == self.num_steps:
            cum_sum, mov_avgs_total, mov_avgs_mal_f, mov_avgs_v = [0], [], [], []
            mov_N = 100
            for i, x in enumerate(self.result_total, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_total = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_total.append(mov_avg_total)

            cum_sum = [0]
            for i, x in enumerate(self.result_mal_f, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_mal_f = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_mal_f.append(mov_avg_mal_f)

            cum_sum = [0]
            for i, x in enumerate(self.result_v, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_v = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_v.append(mov_avg_v)

            # save mv results
            with open('./tmp/results/output_MTD_exp_v1_ft/step_rewards_mv_total.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_total)))
            with open('./tmp/results/output_MTD_exp_v1_ft/step_rewards_mv_mal_f.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_mal_f)))
            with open('./tmp/results/output_MTD_exp_v1_ft/step_rewards_mv_v.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_v)))

            # save raw results
            with open('./tmp/results/output_MTD_exp_v1_ft/step_rewards_total.txt', 'w') as file:
                file.write('\n'.join(map(str, self.result_total)))
            with open('./tmp/results/output_MTD_exp_v1_ft/step_rewards_mal_f.txt', 'w') as file:
                file.write('\n'.join(map(str, self.result_mal_f)))
            with open('./tmp/results/output_MTD_exp_v1_ft/step_rewards_v.txt', 'w') as file:
                file.write('\n'.join(map(str, self.result_v)))

        return self.state, self.reward_total, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        # print('flows_o=', self.num_flows_at_switch)
        # print('routed_num_flows=', self.routed_num_flows)
        # print('data_rate_o=', self.sum_flows_rate_at_switch)
        # print('total_data_rate=', sum(self.sum_flows_rate_at_switch))

    def _compute_observation(self, action):
        next_state_lambda_hat = self.sum_flows_rate_at_switch[-self.num_group:]
        next_state_y = np.multiply(next_state_lambda_hat, action)
        sum_y = np.sum(next_state_y)
        next_state_y_bar = (next_state_y/ sum_y) * self.total_cap_TA
        next_state = np.append(next_state_lambda_hat, next_state_y_bar)
        # print(next_state_lambda_hat)
        # print (action)
        # print (next_state_y_bar)
        # print (np.sum(next_state_y_bar))
        # print (next_state)
        return next_state

    def _compute_reward(self, cur_state, cur_action):
        # experiments
        lambda_hat = cur_state[-self.num_group:]
        y = np.multiply(lambda_hat, cur_action)
        sum_y = np.sum(y)
        y_bar = y * (self.total_cap_TA / sum_y)
        print(y_bar)

        mtd_prob = self.reward_v
        print('mtd_prob: ', mtd_prob)
        self.F_samp, mal_samp, mean_prob_attack, ep_mal = self._change_sampling_rate(y_bar, mtd_prob, self.attacked_groups)
        F_tot = self.num_flows
        self.reward_f = mal_samp / (mal_samp + self.F_samp + 1)
        if ep_mal > 0:
            self.attacked_groups[ep_mal-1] = 1
        print('sampled_f=', self.reward_f)
        print('reward=', mal_samp)
        print('reward_v=', mean_prob_attack)
        print('attacked g=', self.attacked_groups)

        return mal_samp, mal_samp, mean_prob_attack

    def _compute_done(self):
        pass

    # Set topology info.
    def _set_topology(self):
        # Set routing matrix
        routing_mat = []
        f = open(self.topology_path, 'rb')
        while True:
            try:
                routing_mat = pickle.load(f)
            except EOFError:
                break
        f.close()

        # Set flow data rate
        rate_flows, priority_flows, routed_num_flows = self._set_rate_flows(routing_mat)

        # Set traffic matrix
        traffic_mat = [[0 for col in range(self.num_switches)] for row in range(self.num_flows)]
        for j in range(self.num_flows):
            for i in range(self.num_switches):
                traffic_mat[j][i] = rate_flows[j] * routing_mat[j][i]

        # Set hop matrix for random
        # hop_mat = []
        # for j in range(self.num_switches):
        #     hop_jk = []
        #     for k in range(self.num_TA):
        #         hop_jk.append(random.randint(self.min_hop, self.max_hop))
        #     hop_mat.append(hop_jk)

        # Set hop matrix for internet as
        # hop_mat = []
        # for j in range(self.num_switches):
        #     hop_jk = []
        #     for k in range(self.num_TA):
        #         if j < 5 and j >= 0:
        #             hop_jk.append(1)
        #         if j < 10 and j >= 5:
        #             hop_jk.append(2)
        #         if j < 15 and j >= 10:
        #             hop_jk.append(3)
        #         if j < 20 and j >= 15:
        #             hop_jk.append(4)
        #         if j < 25 and j >= 20:
        #             hop_jk.append(5)
        #         if j < 30 and j >= 25:
        #             hop_jk.append(6)
        #     hop_mat.append(hop_jk)

        # Set hop matrix for fattree
        hop_mat = []
        for j in range(self.num_switches):
            hop_jk = []
            for k in range(self.num_TA):
                if j < 2 and j >= 0:
                    hop_jk.append(1)
                if j < 6 and j >= 2:
                    hop_jk.append(2)
                if j < 10 and j >=6:
                    hop_jk.append(3)
            hop_mat.append(hop_jk)

        return routing_mat, traffic_mat, rate_flows, priority_flows, routed_num_flows, hop_mat

    def _set_rate_flows(self, routing_mat):
        rate_flows = []
        priority_flows = []
        routed_num_flows = 0
        for j in range(self.num_flows):
            # Calculate flow data rate
            if sum(routing_mat[j]) is 0:
                rate_flows.append(0)
                priority_flows.append(0)
            if sum(routing_mat[j]) != 0:
                rate_flows.append(random.randint(self.min_flow_rate, self.max_flow_rate))
                priority_flows.append(random.randint(1, 5))
                routed_num_flows += 1

        return rate_flows, priority_flows, routed_num_flows

    # Compute combination
    def _compute_nCk(self, n, k):
        numerator = 1
        denominator = 1
        k = min(n - k, k)
        for i in range(1, k + 1):
            denominator *= i
            numerator *= n + 1 - i
        return numerator // denominator

    # def _compute_nCk_lib(self, n, k):
    #     n_fac = math.factorial(n)
    #     n_k_fac = math.factorial(n - k)
    #     k_fac = math.factorial(k)
    #
    #     return n_fac // n_k_fac // k_fac

    # Compute sampling points using FBC
    def _compute_fbc_samp_points(self):
        fbc_samp_point = []
        routing_m = np.array(self.routing_matrix)
        while (len(fbc_samp_point) != self.num_group):
        #while (routing_m.sum() != 0):
            col_sum = routing_m.sum(axis=0)
            idx_fbc = np.random.choice(np.flatnonzero(col_sum == col_sum.max()))
            if idx_fbc+1 not in fbc_samp_point:
                fbc_samp_point.append(idx_fbc+1)
                routing_m_row = routing_m[:, idx_fbc]
                for i in range(len(routing_m_row)):
                    if routing_m_row[i] == 1:
                        routing_m[i, :] = 0

        return fbc_samp_point

    def _change_sampling_rate(self, samp_rate_o, mtd_prob, attacked_g):
        samp_flows = 0.0
        # Randomly generate malicious flows
        epsilon_mal = np.random.random()
        if epsilon_mal < 0.05:
            self.num_flows = 1255
            cmd = "./div_change_rate.sh "
            cmd += "0 0 0 0 0 0 "
            for i in range(len(samp_rate_o)):
                cmd += str(int(samp_rate_o[i])) + ' '
            cmd += str(epsilon_mal) + ' '
            cmd += str(mtd_prob) + ' '
            cmd += str(attacked_g[0]) + ' '
            cmd += str(attacked_g[1])+ ' '
            cmd += str(attacked_g[2])+ ' '
            cmd += str(attacked_g[3])
            print(cmd)
            subprocess.call(cmd, shell=True)
        else:
            self.num_flows = 1000
            cmd = "./div_change_rate.sh "
            cmd += "0 0 0 0 0 0 "
            for i in range(len(samp_rate_o)):
                cmd += str(int(samp_rate_o[i])) + ' '
            cmd += str(epsilon_mal) + ' '
            cmd += str(mtd_prob) + ' '
            cmd += str(attacked_g[0]) + ' '
            cmd += str(attacked_g[1])+ ' '
            cmd += str(attacked_g[2])+ ' '
            cmd += str(attacked_g[3])
            print(cmd)
            subprocess.call(cmd, shell=True)

        cmd = "scp -P 22 -r wits_controller@172.26.17.82:/home/wits_controller/DRL_sampling/result.txt /home/shkim/PycharmProjects/sdn-sampling/sdn_sampling/envs/TA_result.txt"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        f = open('TA_result.txt', 'r')
        samp_flows = f.readline()
        ids_alerts = f.readline()
        scan_success_prob = f.readline()
        attacked_group = f.readline()
        scan_success_prob = float(scan_success_prob) / 25.0
        print("sampled", samp_flows)
        print("num alerts", ids_alerts)
        print("attack probs", scan_success_prob)
        print("attacked group", attacked_group)
        f.close()

        return int(samp_flows), int(ids_alerts), float(scan_success_prob), int(attacked_group)