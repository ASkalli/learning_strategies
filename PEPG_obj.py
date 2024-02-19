# -*- coding: utf-8 -*-
"""
Based mostly on the oto code from David Ha:

https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
"""

import numpy as np

class PEPG_opt:
    
    def __init__(self, num_params ,pop_size, learning_rate, starting_mu, starting_sigma):
        
        self.pop_size = pop_size + 1 if pop_size % 2 else pop_size
        self.num_params = num_params
        self.mu = np.array(starting_mu).flatten()
        self.batch_size = self.pop_size // 2
        self.batch_reward = np.zeros(2 * self.batch_size)
        self.sigma = np.ones(num_params) * starting_sigma
        self.sigma_init = starting_sigma
        
        # Parameters for the PEPG optimizer
        self.sigma_alpha = 0.30
        self.sigma_decay = 0.999
        self.sigma_limit = 0.01
        self.sigma_max_change = 0.2
        self.learning_rate = learning_rate
        self.learning_rate_decay = 0.99
        self.learning_rate_limit = 0.01
        self.elite_ratio = 0
        self.weight_decay = 0.01
        self.forget_best = True
        self.rank_fitness = True
        self.average_baseline = True
        self.use_elite = False

        self.first_iteration = True
        self.best_reward = None
        self.best_mu = None

    def compute_ranks(self, x):
        #ranks = np.empty_like(x)
        #ranks[x.argsort(axis=0)] = np.arange(len(x))
        #return ranks
        argsorted_x = np.argsort(x,axis=0)
        ranks = np.empty_like(argsorted_x, dtype=float)
        ranks[argsorted_x] = np.arange(len(x))
        return ranks

    def compute_centered_ranks(self, x):
        y = self.compute_ranks(x)
        y = y / (len(x) - 1)
        return y - 0.5

    def ask(self):
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma
        self.epsilon_full = np.vstack((self.epsilon, -self.epsilon))
        if self.average_baseline:
            return self.mu + self.epsilon_full
        else:
            return np.vstack((np.zeros(self.num_params), self.mu + self.epsilon_full))

    def tell(self, reward_table_result):
        reward_table = np.array(reward_table_result).flatten()
        if self.rank_fitness:
            reward_table = self.compute_centered_ranks(reward_table)

        if self.average_baseline:
            b = np.mean(reward_table)
        else:
            b = reward_table[0]

        reward = reward_table[0:] if self.average_baseline else reward_table[1:]

        #best_reward_index = np.argmin(reward) if self.rank_fitness else np.argmax(reward)
        #best_reward = reward[best_reward_index]
        best_reward_index = np.argmin(reward)
        best_reward = reward[best_reward_index]

        if best_reward > b or self.average_baseline:
            best_mu = self.mu + self.epsilon_full[best_reward_index]
        else:
            best_mu = self.mu

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_iteration:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_iteration = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        elif self.forget_best or self.curr_best_reward > self.best_reward:
            self.best_mu = best_mu
            self.best_reward = self.curr_best_reward

        # Update mu
        if self.use_elite:
            elite_indices = np.argsort(reward_table_result)[:int(self.elite_ratio * len(reward_table_result))]
            self.mu += np.mean(self.epsilon_full[elite_indices], axis=0)
        else:
            if self.rank_fitness:
                reward_table = self.compute_centered_ranks(np.array(reward_table_result).flatten())
            rT = reward_table[:self.batch_size] - reward_table[self.batch_size:]
            change_mu = np.dot(rT, self.epsilon_full[:self.batch_size])
            self.mu -= self.learning_rate * change_mu
    
        # Update sigma
        if self.sigma_alpha > 0:
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = np.std(reward_table_result)
            S = (self.epsilon ** 2 - self.sigma ** 2) / self.sigma
            reward_avg = (reward_table[:self.batch_size] + reward_table[self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = np.dot(rS.T, S) / (2 * self.batch_size * stdev_reward)
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.clip(change_sigma, -self.sigma_max_change * self.sigma, self.sigma_max_change * self.sigma)
            self.sigma -= change_sigma.squeeze()
    
        # Apply sigma decay
        if self.sigma_decay < 1:
            self.sigma *= self.sigma_decay
            self.sigma = np.maximum(self.sigma, self.sigma_limit)
    
        # Apply learning rate decay
        if self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit:
            self.learning_rate *= self.learning_rate_decay
    
        return 0 


# Example usage:
# pe = PEPG_Optimization(pop_size, num_params, learning_rate, starting_mu, starting_sigma)
# solutions = pe.ask()
# pe.tell(reward_table_result)
