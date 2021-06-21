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

class SdnSamplingEnv_MTD_v1(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        # set result var.
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
        self.num_steps = 30000
        # self.topology_path = './tmp/topo/routing_matrix_30_50000.txt'
        # self.num_switches = 30
        # self.num_flows = 50000
        self.topology_path = './tmp/topo/routing_matrix_10_10000.txt'
        self.num_switches = 10
        self.num_flows = 10000
        self.num_TA = 4
        self.num_group = 4
        self.total_cap_TA = 400000.0
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
        # Randomly change flow data rate
        epsilon = np.random.random()
        if epsilon < 0.1:
            self.routing_matrix, self.traffic_matrix, self.rate_flows, self.priority_flows, self.routed_num_flows, tmp_hop_mat = self._set_topology()
            # self.mal_flows = self._set_mal_flows()

            # number of flows at switch
            tmp_rm = np.asmatrix(self.routing_matrix)
            self.num_flows_at_switch = np.squeeze(
                np.asarray(tmp_rm.sum(axis=0))
            )
            # print('|F^tot|=', self.num_flows_at_switch)
            # print('routed_num_flows=', self.routed_num_flows)

            # data rate at switch
            tmp_tm = np.asmatrix(self.traffic_matrix)
            self.sum_flows_rate_at_switch = np.squeeze(
                np.asarray(tmp_tm.sum(axis=0))
            )
            #print('data_rate_o=', self.sum_flows_rate_at_switch)
            #print('total_data_rate=', sum(self.sum_flows_rate_at_switch))
            #print('--------------------------------------')
        else:
            pass

        # Scale action
        # print('selected action:', action)

        # Get next state according to the current action
        self.state = self._compute_observation(action)
        # print('next state:', self.state)

        # Get reward with current state and action
        self.reward_total = self._compute_reward(self.state, action)
        #print('reward at st at:', reward)

        # Set time-step per episode
        done = True
        #done = bool(self.reward_total >= 1)
        info = {}

        # save rewards
        self.result_total.append(self.reward_total)
        # print(self.result_f)
        # print(self.result_u)
        # print(self.result_v)
        # print(self.result_total)
        # print('-----------------------')
        # print('reward_total:', self.reward_total)

        if len(self.result_f) == self.num_steps:
            cum_sum, mov_avgs_total = [0], []
            mov_N = 100
            for i, x in enumerate(self.result_total, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_total = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_total.append(mov_avg_total)

            with open('./tmp/results/output_MTD_v1_ddpg_as/step_rewards_total.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_total)))

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
        next_state_y_bar = next_state_y * (self.total_cap_TA / sum_y)
        next_state = np.append(next_state_lambda_hat, next_state_y_bar)
        # print(next_state_lambda_hat)
        # print (action)
        # print (next_state_y_bar)
        # print (np.sum(next_state_y_bar))
        # print (next_state)
        return next_state

    def _compute_reward(self, cur_state, cur_action):
        lambda_hat = cur_state[-self.num_group:]
        y = np.multiply(lambda_hat, cur_action)
        sum_y = np.sum(y)
        y_bar = y * (self.total_cap_TA / sum_y)

        self.F_samp = 0
        samp_mat = []
        samp_mat = y_bar / self.sum_flows_rate_at_switch[-self.num_group:]
        samp_mat = np.append(np.zeros(self.num_switches - self.num_group), samp_mat)

        for i in range (self.num_switches):
            if samp_mat[i] > 1:
                samp_mat[i] = 1

        # print (samp_mat)

        sampled_flows = []
        sampled_flows = np.matmul(self.routing_matrix, samp_mat)

        for flow_index in range(self.num_flows):
            if sampled_flows[flow_index] > 0:
                self.F_samp += 1
        # print('sampled=', self.F_samp)

        num_mal = math.ceil(self.num_flows * self.malicious_rate)
        alerts = 0
        mal_samp = 0
        mal_samp_tmp = 0
        for flow_index in range(self.num_flows - num_mal, self.num_flows):
            if sampled_flows[flow_index] >= .5:
                mal_samp_tmp += 1
            alerts += (sampled_flows[flow_index] / self.priority_flows[flow_index])
        mal_samp = mal_samp_tmp / num_mal
        print('mal_sampled=', mal_samp)
        print('reward=', alerts)

        # capture_failure = np.ones(self.num_flows)
        # for flow_index in range(self.num_flows):
        #     if self.rate_flows[flow_index] != 0:
        #         for i in range(self.num_switches):
        #             if self.routing_matrix[flow_index][i] != 0:
        #                 n1 = int(self.sum_flows_rate_at_switch[i] - self.rate_flows[flow_index])
        #                 k1 = int(samp_mat[i])
        #                 if n1 < k1:
        #                     capture_failure[flow_index] = 0
        #                 else:
        #                     a = self._compute_nCk(n1, k1)
        #                     n2 = int(self.sum_flows_rate_at_switch[i])
        #                     k2 = int(samp_mat[i])
        #                     b = self._compute_nCk(n2, k2)
        #                     capture_failure[flow_index] *= a/b
        #             else:
        #                 pass
        #     else:
        #         capture_failure[flow_index] = 0
        # print(capture_failure)
        # #
        # mal_samp = 0
        # mal_samp_tmp = []
        # num_mal = math.ceil(self.num_flows * self.malicious_rate)
        # for flow_index in range(self.num_flows - num_mal, self.num_flows):
        #     #if capture_failure[flow_index] < 1:
        #     #    mal_samp += 1
        #     mal_samp_tmp.append(capture_failure[flow_index])
        #     # print(self.routing_matrix[flow_index])
        # mal_samp = (np.mean(mal_samp_tmp))
        # print('mal_sampled=', mal_samp)

        return alerts

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
        hop_mat = []
        for j in range(self.num_switches):
            hop_jk = []
            for k in range(self.num_TA):
                if j < 5 and j >= 0:
                    hop_jk.append(1)
                if j < 10 and j >= 5:
                    hop_jk.append(2)
                if j < 15 and j >= 10:
                    hop_jk.append(3)
                if j < 20 and j >= 15:
                    hop_jk.append(4)
                if j < 25 and j >= 20:
                    hop_jk.append(5)
                if j < 30 and j >= 25:
                    hop_jk.append(6)
            hop_mat.append(hop_jk)

        # Set hop matrix for fattree
        # hop_mat = []
        # for j in range(self.num_switches):
        #     hop_jk = []
        #     for k in range(self.num_TA):
        #         if j < 2 and j >= 0:
        #             hop_jk.append(1)
        #         if j < 6 and j >= 2:
        #             hop_jk.append(2)
        #         if j < 10 and j >=6:
        #             hop_jk.append(3)
        #     hop_mat.append(hop_jk)

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