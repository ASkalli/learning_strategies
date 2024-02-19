# -*- coding: utf-8 -*-
"""
Based on the purecmaes matlab code from Nikolaus Hansen:

https://cma-es.github.io/cmaes_sourcecode_page.html#matlab

"""

import numpy as np

class CMA_opt:
    def __init__(self, N_dim, pop_size, select_pop, sigma_init, mean_init):
        # Ensure mean_init is a numpy array and has the correct shape
        if not isinstance(mean_init, np.ndarray):
            raise ValueError("mean_init must be a numpy array")
        if mean_init.shape not in [(N_dim,), (N_dim, 1)]:
            raise ValueError(f"mean_init must be of shape ({N_dim},) or ({N_dim}, 1)")

        # Reshape mean_init to a column vector if necessary
        self.xmean = mean_init.reshape(N_dim, 1) if mean_init.ndim == 1 else mean_init

        self.N_dim = N_dim
        self.sigma = sigma_init
        self.pop_size = pop_size
        self.select_pop = select_pop

        # Calculate weights for the top 'select_pop' individuals with a power law
        self.weights = np.array([np.log(select_pop + 0.5) - np.log(i) for i in range(1, select_pop + 1)])
        #calculate weights with a linear scaling
        #self.weights = np.linspace(1, 0.01, select_pop)
        #constant weights
        #self.weights = np.ones(select_pop)
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights


        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / self.N_dim) / (self.N_dim + 4 + 2 * self.mueff / self.N_dim)
        self.cs = (self.mueff + 2) / (self.N_dim + self.mueff + 5)
        self.c1 = 2 / ((self.N_dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.N_dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N_dim + 1)) - 1) + self.cs

        # Initialize evolution paths and covariance matrix
        self.pc = np.zeros((N_dim, 1))
        self.ps = np.zeros((N_dim, 1))
        self.B = np.eye(N_dim)
        self.D = np.ones([N_dim, 1])
        self.C = self.B @ np.diag((self.D**2).flatten()) @ self.B.T
        self.invsqrtC = self.B @ np.diag((self.D**-1).flatten()) @ self.B.T
        self.eigeneval = 0
        self.chiN = N_dim**0.5 * (1 - 1/(4*N_dim) + 1/(21*N_dim**2))

        self.epsilon = 1e-8  # A small value to ensure numerical stability
        self.sigma_max = 1e10  # Upper bound for sigma
        self.sigma_min = 1e-10  # Lower bound for sigma
        self.eigen_update_frequency = int(N_dim / 10)
        #self.eigen_update_frequency = 1
        
        self.population = None
        self.counteval = 0
        self.best_solution = None
        self.best_fitness = None
        
    def ask(self):
        # Generate a new population
        self.population = self.xmean + self.sigma * (self.B @ (self.D * np.random.randn(self.N_dim, self.pop_size)))
        
        # Validation check for population generation
        if self.population.shape != (self.N_dim, self.pop_size):
            raise ValueError("Generated population size does not match the specified dimensions")

        return self.population

    def tell(self, reward_table):
        # Validation check for reward_table
        if not isinstance(reward_table, (list, np.ndarray)) or len(reward_table) != self.pop_size:
            raise ValueError("reward_table must be a list or numpy array with length equal to the population size")

        arfitness = np.array(reward_table)
        arindex = np.argsort(arfitness,axis=0)
        xold = self.xmean.copy()

        # Update mean
        self.xmean = np.dot(self.population[:, arindex[:self.select_pop]].squeeze(), self.weights[:, np.newaxis])

        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (self.xmean - xold) / self.sigma

        hsig = np.sum(self.ps**2) / (1 - (1 - self.cs)**(2 * self.counteval / self.pop_size) + self.epsilon) / self.N_dim < 2 + 4 / (self.N_dim + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.xmean - xold) / self.sigma

        # Adapt covariance matrix C with numerical stability enhancement
        artmp = (1 / self.sigma) * (self.population[:, arindex[:self.select_pop]].squeeze() - np.tile(xold, (1, self.select_pop)))
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights) @ artmp.T
        np.fill_diagonal(self.C, self.C.diagonal() + self.epsilon)

        # Adapt step size sigma with boundary checks
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = min(max(self.sigma, self.sigma_min), self.sigma_max)

        # Update eigen decomposition of the covariance matrix C less frequently
        if self.counteval % self.eigen_update_frequency == 0:
            self.C = (self.C + self.C.T) / 2  # Ensure C is symmetric
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D[:,np.newaxis])
            #self.D = np.sqrt(np.diag(self.D))
            self.invsqrtC = self.B @ np.diag((self.D**-1).flatten()) @ self.B.T

        # Log the best individual solution and its fitness
        current_best_index = np.argmin(reward_table)
        current_best_solution = self.population[:, current_best_index]
        current_best_fitness = reward_table[current_best_index]

        if self.best_fitness is None or current_best_fitness < self.best_fitness:
            self.best_solution = current_best_solution.copy()
            self.best_fitness = current_best_fitness

        self.counteval += 1