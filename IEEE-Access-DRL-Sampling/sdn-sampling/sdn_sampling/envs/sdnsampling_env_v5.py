import itertools
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
from numpy.linalg import inv

class SdnSamplingEnv_v5(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        # set result var.
        self.routing_candidate = 15
        self.malicious_rate = 0.02
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
        self.num_steps = 50000
        self.topology_path = './tmp/topo/routing_matrix_30_50000.txt'
        self.num_switches = 30
        self.num_flows = 50000
        # self.topology_path = './tmp/topo/routing_matrix_10_500.txt'
        # self.num_switches = 10
        # self.num_flows = 500
        self.num_TA = 5
        self.num_samp = 10
        self.total_cap_TA = 50000.0
        self.r = 0.5

        self.min_flow_rate = 1.0
        self.max_flow_rate = 20.0
        self.min_sampled_flows = 0
        self.max_sampled_flows = self.num_flows

        self.min_hop = 1
        self.max_hop = 6

        self.routing_matrix, \
        self.traffic_matrix, \
        self.rate_flows, \
        self.routed_num_flows, \
        self.hop_mat = self._set_topology()
        #self.mal_flows = self._set_mal_flows()

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
        # print('data_rate_o=', self.sum_flows_rate_at_switch)
        # print('total_data_rate=', sum(self.sum_flows_rate_at_switch))
        # print('hop matrix:', self.hop_mat)

        # Set capacity of traffic analyzers
        self.cap_TA = self.total_cap_TA / self.num_TA

        # Set state space s_t
        self.observation_space = spaces.Box(low=1, \
                                            high=self.num_switches, \
                                            shape=(self.num_samp * 2,), \
                                            dtype=np.int)

        # Set action space a_t
        self.action_space = spaces.Box(low=np.array([1, 1, 0.0]), \
                                    high=np.array([self.num_switches+0.5, self.num_TA+0.5, 0.99]), \
                                    dtype=np.float)

        # Set initial state
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        # Set initial state
        while True:
            self.state = self.observation_space.sample()
            init_tmp_samp_point = []
            init_set_samp_point = []

            for j in range(0, len(self.state), 2):
                init_tmp_samp_point.append(self.state[j])

            for value in init_tmp_samp_point:
                if value not in init_set_samp_point:
                    init_set_samp_point.append(value)

            for k in range(1, len(self.state), 2):
                if(self.state[k]) > self.num_TA:
                    self.state[k] = random.randrange(1, self.num_TA+1)

            if len(init_set_samp_point) == self.num_samp:
                break

        done = False

        return np.array(self.state)

    def step(self, action):
        # Randomly change flow data rate
        epsilon = np.random.random()
        if epsilon < 0.1:
            self.routing_matrix, self.traffic_matrix, self.rate_flows, self.routed_num_flows, tmp_hop_mat = self._set_topology()
            # self.mal_flows = self._set_mal_flows()

            # number of flows at switch
            tmp_rm = np.asmatrix(self.routing_matrix)
            self.num_flows_at_switch = np.squeeze(
                np.asarray(tmp_rm.sum(axis=0))
            )
            #print('|F^tot|=', self.num_flows_at_switch)
            #print('routed_num_flows=', self.routed_num_flows)

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

        # Select sampling point using FBC and perform greedy TA selection
        fbc_points = self._compute_fbc_samp_points()
        print('centrality:', fbc_points)
        greedy_TA_selection = self._compute_greedy_TA_selection(fbc_points)
        print('TA selection:', greedy_TA_selection)
        self.state = self._compute_fbc_observation(fbc_points, greedy_TA_selection)

        # Get next state according to the current action
        #self.state = self._compute_observation(action)

        #print('next state:', self.state)

        # Get reward with current state and action
        #self.res_util_TAs, self.reward_total = self._compute_reward(self.state)
        #print('reward at (st, at):', self.reward_total)

        # Get fbc_reward
        self.mal_samp, self.res_util_TAs, self.mean_overhead, self.reward_total = self._compute_reward_fbc_rate_prop_geedy(self.state)

        # Set time-step per episode
        done = bool(self.reward_total >= 1)
        #done = True
        info = {}

        # save rewards
        # result_F_samp.append(self.F_samp)
        self.result_f.append(self.reward_f)
        self.result_mal_f.append(self.mal_samp)
        # result_res_util_TAs.append(self.res_util_TAs)
        self.result_u.append(self.reward_u)
        self.result_mean_overhead.append(self.mean_overhead)
        self.result_v.append(self.reward_v)
        self.result_total.append(self.reward_total)
        # print(self.result_f)
        # print(self.result_u)
        # print(self.result_v)
        # print(self.result_total)
        print('-----------------------')
        print('mal_sampled', self.mal_samp)
        print('num_sampled:', self.F_samp)
        print('reward_f:', self.reward_f)
        print('util_TAs:', self.res_util_TAs)
        print('reward_u:', self.reward_u)
        print('mean_overheads:', self.mean_overhead)
        print('reward_v:', self.reward_v)
        print('reward_total:', self.reward_total)

        if len(self.result_f) == self.num_steps:
            cum_sum, mov_avgs_f, mov_avgs_mal_f, mov_avgs_u, mov_avgs_v, mov_avgs_mean_overhead, mov_avgs_total = [0], [], [], [], [], [], []
            mov_N = 1
            for i, x in enumerate(self.result_f, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_f = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_f.append(mov_avg_f)

            cum_sum = [0]
            for i, x in enumerate(self.result_mal_f, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_mal_f = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_mal_f.append(mov_avg_mal_f)

            cum_sum = [0]
            for i, x in enumerate(self.result_u, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_u = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_u.append(mov_avg_u)

            cum_sum = [0]
            for i, x in enumerate(self.result_v, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_v = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_v.append(mov_avg_v)

            cum_sum = [0]
            for i, x in enumerate(self.result_mean_overhead, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_mean_overhead = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_mean_overhead.append(mov_avg_mean_overhead)

            cum_sum = [0]
            for i, x in enumerate(self.result_total, 1):
                cum_sum.append(cum_sum[i - 1] + x)
                if i >= mov_N:
                    mov_avg_total = (cum_sum[i] - cum_sum[i - mov_N]) / mov_N
                    mov_avgs_total.append(mov_avg_total)

            with open('./tmp/results/output_FBC/step_rewards_f.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_f)))
            with open('./tmp/results/output_FBC/step_rewards_mal_f.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_mal_f)))
            with open('./tmp/results/output_FBC/step_rewards_u.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_u)))
            with open('./tmp/results/output_FBC/step_rewards_v.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_v)))
            with open('./tmp/results/output_FBC/step_mean_overhead.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_mean_overhead)))
            with open('./tmp/results/output_FBC/step_rewards_total.txt', 'w') as file:
                file.write('\n'.join(map(str, mov_avgs_total)))

        return self.state, self.reward_total, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        #print("render")
        print('flows_o=', self.num_flows_at_switch)
        print('routed_num_flows=', self.routed_num_flows)
        print('data_rate_o=', self.sum_flows_rate_at_switch)
        print('total_data_rate=', sum(self.sum_flows_rate_at_switch))
        print('sampling_reduction_r=', self.r)
        print('hop matrix:', self.hop_mat)
        self._set_statistics()

    def _set_statistics(self):
        F_samp = self.F_samp
        reward_f = self.reward_f
        res_util_TAs = self.res_util_TAs
        reward_u = self.reward_u
        mean_overhead = self.mean_overhead
        reward_v = self.reward_v
        reward_total = self.reward_total
        print('num_sampled:', F_samp)
        print('reward_f:', reward_f)
        print('util_TAs:', res_util_TAs)
        print('reward_u:', reward_u)
        print('mean_overheads:', mean_overhead)
        print('reward_v:', reward_v)
        print('reward_total:', reward_total)

        # step_results = []
        # step_results.append(F_samp)
        # step_results.append(reward_f)
        # step_results.append(res_util_TAs)
        # step_results.append(reward_u)
        # step_results.append(reward_total)
        # with open('./output1/step_rewards_1.txt', 'w') as file:
        #     file.write('\n'.join(map(str, step_results)))

    def _compute_observation(self, action):
        tmp_state = []
        for j in range(0, len(self.state), 2):
            if self.state[j] != action[0] and len(tmp_state) < (self.num_samp - 1) * 2:
                tmp_state.append(self.state[j])
                tmp_state.append(self.state[j + 1])
        next_state = np.append(action[0], action[1])
        next_state = np.append(next_state, tmp_state)

        return next_state

    # Compute reward for BC sampling points, rate-prop sampling rates, and greedy TA selection
    def _compute_reward_fbc_rate_prop_geedy(self, cur_state):
        samp_point_p = []
        selected_TA_d = []

        for j in range(0, len(cur_state), 2):
            samp_point_p.append(cur_state[j])
        # print('samp_point_p=', samp_point_p)

        for k in range(1, len(cur_state), 2):
            selected_TA_d.append((cur_state[k]))
        # print('seleted_TA_d=', selected_TA_d)

        # Calculate sampled flow reward r^f
        sum_sw_rate = 0
        for j in range(len(samp_point_p)):
            sum_sw_rate += self.sum_flows_rate_at_switch[samp_point_p[j] - 1]
        samp_rate_x = []
        for j in range(len(samp_point_p)):
            samp_rate_x.append(self.sum_flows_rate_at_switch[samp_point_p[j] - 1] * self.total_cap_TA / sum_sw_rate)
        #print('samp_rate_x=', samp_rate_x)
        #print(np.sum(samp_rate_x))
        #print(self.sum_flows_rate_at_switch)

        samp_rate_o = np.zeros(self.num_switches)
        tmp_samp_rate_o = samp_rate_o

        for j in range(len(samp_point_p)):
            tmp_samp_rate_o[int(samp_point_p[j] - 1)] = samp_rate_x[j]
        # print('tmp_samp_rate_o=', tmp_samp_rate_o)

        samp_rate_o = tmp_samp_rate_o
        # for i in range(self.num_switches):
        #     if tmp_samp_rate_o[i] < self.sum_flows_rate_at_switch[i]:
        #         samp_rate_o[i] = tmp_samp_rate_o[i]
        #     else:
        #         samp_rate_o[i] = self.sum_flows_rate_at_switch[i]
        # print('sum_samp_rate_o=', np.sum(samp_rate_o))

        # Calculate rate-prop sampling rate
        # sum_fw_rate = np.sum(self.rate_flows)
        # b = []
        # for i in range (self.num_flows):
        #     b.append((self.total_cap_TA / sum_fw_rate) * self.rate_flows[i])
        # print (np.sum(b))
        # A = self.traffic_matrix
        # mat_A = np.asmatrix(A)
        # mat_A_T = mat_A.transpose()
        # mat_A_T_A = np.matmul(mat_A_T, mat_A)
        # inverse_mat_A_T_A = inv(mat_A_T_A)
        # tmp = np.matmul(inverse_mat_A_T_A, mat_A_T)
        # tmp_samp_rate_o = np.matmul(tmp, b)
        # tmp_samp_rate_o = np.array(tmp_samp_rate_o).reshape(-1).tolist()
        # for i in range(self.num_switches):
        #     if tmp_samp_rate_o[i] > 1:
        #         tmp_samp_rate_o[i] = 1
        #     if tmp_samp_rate_o[i] < 0:
        #         tmp_samp_rate_o[i] = 0
        # # print(tmp_samp_rate_o)
        # samp_rate_o = []
        # for i in range(self.num_switches):
        #     samp_rate_o.append(tmp_samp_rate_o[i] * self.sum_flows_rate_at_switch[i])

        # for i in range(self.num_switches):
        #     samp_rate_o[i] = 1000
        # print('samp_rate_o=', samp_rate_o)

        fbc_points1 = self._compute_fbc_samp_points()
        fbc_candidate = fbc_points1[: self.routing_candidate]
        num_mal = math.ceil(self.num_flows * self.malicious_rate)
        print(fbc_candidate)

        for i in range(int(self.num_flows-num_mal), self.num_flows):
            self.routing_matrix[i] = [0 for j in range (self.num_switches)]
            while np.sum(self.routing_matrix[i]) < 1:
                j = np.random.randint(1, self.num_switches)
                if j not in fbc_candidate:
                    self.routing_matrix[i][j-1] = 1
        #     print(str(self.routing_matrix[i]))
        # print ('------------------------')

        # capture_failure = np.ones(self.num_flows)
        # for flow_index in range(self.num_flows):
        #     if self.rate_flows[flow_index] != 0:
        #         for i in range(self.num_switches):
        #             if self.routing_matrix[flow_index][i] != 0:
        #                 n1 = int(self.sum_flows_rate_at_switch[i] - self.rate_flows[flow_index])
        #                 k1 = int(samp_rate_o[i])
        #                 if n1 < k1:
        #                     capture_failure[flow_index] = 0
        #                 else:
        #                     a = self._compute_nCk(n1, k1)
        #                     n2 = int(self.sum_flows_rate_at_switch[i])
        #                     k2 = int(samp_rate_o[i])
        #                     b = self._compute_nCk(n2, k2)
        #                     capture_failure[flow_index] *= a / b
        #             else:
        #                 pass
        #     else:
        #         capture_failure[flow_index] = 0
        # # #print(np.mean(capture_failure))
        # #
        # mal_samp = 0
        # mal_samp_tmp = []
        # num_mal = math.ceil(self.num_flows * self.malicious_rate)
        # for flow_index in range(self.num_flows - num_mal, self.num_flows):
        #     #if capture_failure[flow_index] < 1:
        #         #mal_samp += 1
        #     mal_samp_tmp.append(capture_failure[flow_index])
        #     # print(self.routing_matrix[flow_index])
        # mal_samp = (np.mean(mal_samp_tmp))
        # print('mal_sampled=', mal_samp)
        #
        # self.F_samp = self.routed_num_flows * (1 - np.mean(capture_failure))

        self.F_samp = 0
        samp_mat = []
        samp_mat = samp_rate_o / self.sum_flows_rate_at_switch
        sampled_flows = []
        sampled_flows = np.matmul(self.routing_matrix, samp_mat)
        for flow_index in range(self.num_flows):
            if sampled_flows[flow_index] > 0:
                self.F_samp += 1

        mal_samp_tmp = 0
        mal_samp = 0
        for flow_index in range(self.num_flows - num_mal, self.num_flows):
            if sampled_flows[flow_index] > 0:
                mal_samp_tmp += 1
        mal_samp = mal_samp_tmp / num_mal
        print('mal_sampled=', mal_samp)

        # self.F_samp = 0
        # for j in range(self.num_flows):
        #     if capture_failure[j] != 1:
        #         self.F_samp += 1
        F_tot = self.routed_num_flows
        self.reward_f = self.F_samp / F_tot
        # self.reward_f = (1-np.mean(capture_failure))
        # print('data_rate_o=', self.sum_flows_rate_at_switch)
        # print('samp_rate_o=', samp_rate_o)
        # print('num_captured_flows=', F_samp)
        # print('reward^f=', reward_f)

        # Calculate TA's utilization reward r^u
        res_group_g = np.zeros(self.num_TA)
        for k in range(1, len(cur_state), 2):
            # print(cur_state[k])
            # print(samp_rate_o[cur_state[k-1]-1])
            res_group_g[int(cur_state[k] - 1)] += samp_rate_o[int(cur_state[k - 1] - 1)]
        # print('res_group_g=', res_group_g)

        # Set capacity of traffic analyzers
        # cap_TA = self.total_cap_TA/self.num_TA
        res_util_u = []
        for k in range(self.num_TA):
            # if res_group_g[k] < self.cap_TA:
            #     res_util_u.append(res_group_g[k] / self.cap_TA)
            # else:
            #     res_util_u.append(1)
            res_util_u.append(res_group_g[k] / self.cap_TA)
        # print('res_util_u=', res_util_u)

        jains_n = np.sum(res_util_u) ** 2
        jains_d = 0
        for value in res_util_u:
            jains_d += value ** 2
        self.reward_u = jains_n / (self.num_TA * jains_d)
        # print('reward^u=', reward_u)

        # Calculate penalty r^v
        overhead_v = []
        penalty_v = []
        total_data_rate = sum(self.sum_flows_rate_at_switch)
        for k in range(self.num_TA):
            inner_overhead_v = []
            for j in range(0, len(cur_state), 2):
                if k+1 == cur_state[j+1]:
                    inner_overhead_v.append(self.hop_mat[int(cur_state[j]) - 1][int(k)] * samp_rate_o[int(cur_state[j]) - 1])
            overhead_v.append(np.sum(inner_overhead_v))
        # print(overhead_v)
        max_ov = self.max_hop * self.total_cap_TA
        min_ov = self.total_cap_TA
        # for k in range(self.num_TA):
            # if self.num_TA == 1:
            #    penalty_v.append(0)
            # else:
            #   penalty_v.append((overhead_v[k] - np.min(overhead_v)) / (np.max(overhead_v) - np.min(overhead_v)))
        penalty_v = abs(np.sum(overhead_v) - min_ov) / (max_ov - min_ov)

        # print(penalty_v)
        # self.reward_v = np.mean(penalty_v)
        self.reward_v = penalty_v
        mean_overhead = np.sum(overhead_v)
        # print (overhead_v)
        # print(self.reward_v)
        # print(overhead_d)
        # print(total_data_rate)

        # Calculate total reward R
        # reward_exp = exp(self.reward_f * self.reward_u)
        reward_total = (self.reward_f * self.reward_u) - self.reward_v
        return mal_samp, res_util_u, mean_overhead, reward_total

    # Compute sampling points using FBC
    def _compute_fbc_samp_points(self):
        fbc_samp_point = []
        routing_m = np.array(self.routing_matrix)
        while (len(fbc_samp_point) != self.num_samp):
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

    # Compute greedy TA selection according to the FBC sampling point selection
    def _compute_greedy_TA_selection(self, fbc_points):
        cap_TAs = []
        fbc_rate = []

        for j in range (len(fbc_points)):
            fbc_rate.append(self.sum_flows_rate_at_switch[fbc_points[j]-1])
        # print('fbc_rate:', fbc_rate)

        for k in range(self.num_TA):
            cap_TAs.append(self.cap_TA)
        # print(cap_TAs)

        # Calculate sampled flow reward r^f
        sum_sw_rate = 0
        for j in range(len(fbc_points)):
            sum_sw_rate += self.sum_flows_rate_at_switch[fbc_points[j] - 1]
        samp_rate_x = []
        for j in range(len(fbc_points)):
            samp_rate_x.append(self.sum_flows_rate_at_switch[fbc_points[j] - 1] * self.total_cap_TA / sum_sw_rate)
        # print('samp_rate_x=', samp_rate_x)

        samp_rate_o = np.zeros(self.num_switches)
        tmp_samp_rate_o = samp_rate_o

        for j in range(len(fbc_points)):
            tmp_samp_rate_o[int(fbc_points[j] - 1)] = samp_rate_x[j]
        # print('tmp_samp_rate_o=', tmp_samp_rate_o)

        for i in range(self.num_switches):
            if tmp_samp_rate_o[i] < self.sum_flows_rate_at_switch[i]:
                samp_rate_o[i] = tmp_samp_rate_o[i]
            else:
                samp_rate_o[i] = self.sum_flows_rate_at_switch[i]

        overhead_v = [[0 for col in range(self.num_TA)] for row in range(self.num_switches)]
        for j in fbc_points:
            for k in range(self.num_TA):
                overhead_v[j-1][k] = (self.hop_mat[int(j) - 1][int(k)] * samp_rate_o[int(j) - 1])
        # print (overhead_v)

        selected_TA = []
        for j in fbc_points:
            idx_TA = cap_TAs.index(max(cap_TAs))
            if cap_TAs[idx_TA] > 0:
                selected_TA.append(idx_TA+1)
                cap_TAs[idx_TA] -= samp_rate_o[j-1]
                # print (cap_TAs)

        # print (selected_TA)

        return selected_TA

    # Compute state according to the fbc and greedy method
    def _compute_fbc_observation(self, fbc_points, greedy_TA_selection):
        fbc_state = []
        for j in range(len(fbc_points)):
            fbc_state.append(fbc_points[j])
            fbc_state.append(greedy_TA_selection[j])
        #print(fbc_state)

        return fbc_state

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
        rate_flows, routed_num_flows = self._set_rate_flows(routing_mat)

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
                    hop_jk.append(2)
                if j < 10 and j >= 5:
                    hop_jk.append(3)
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

        return routing_mat, traffic_mat, rate_flows, routed_num_flows, hop_mat

    def _set_rate_flows(self, routing_mat):
        rate_flows = []
        routed_num_flows = 0
        for j in range(self.num_flows):
            # Calculate flow data rate
            if sum(routing_mat[j]) is 0:
                rate_flows.append(0)
            if sum(routing_mat[j]) != 0:
                rate_flows.append(random.randint(self.min_flow_rate, self.max_flow_rate))
                routed_num_flows += 1

        return rate_flows, routed_num_flows

    # Compute combination
    def _compute_nCk(self, n, k):
        numerator = 1
        denominator = 1
        k = min(n - k, k)
        for i in range(1, k + 1):
            denominator *= i
            numerator *= n + 1 - i
        return numerator // denominator
