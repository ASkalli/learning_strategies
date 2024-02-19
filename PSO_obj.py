"""
This is a simple implementation of the Particle Swarm Optimization algorithm.

"""

import numpy as np

class PSO_opt:
    def __init__(self, X_init, V_init, params):
        self.X = X_init  # Position vector for the particles
        self.V = V_init  # Velocity vector for the particles
        self.p_best = np.array(X_init)  # Best position found by each particle, this is a vector
        self.N_population = X_init.shape[1]
        self.N_dim = X_init.shape[0]
        self.w_inertia = params['w'] # inertia basically keeps the particle going along the same trajectory
        self.c_1 = params['c_1'] # cognitive coefficient influences the search of the individual particle
        self.c_2 = params['c_2'] # social coefficient influences the particles to go towards the global best
        self.g_best = np.zeros((self.N_dim, 1))  # Best position ever found by the swarm
        self.best_reward = np.inf
        self.reward_history = []
        self.curr_reward = np.full((self.N_population,1), np.inf)
        self.Vmax = params['Vmax'] # max velocity to make the particles relax a bit ...
        # boundaries of the search space, usefull for hardware weights
        self.upper_bound = params['upper_bound']
        self.lower_bound = params['lower_bound']
        #random values for r_1 and r_2
        self.r_1 = np.random.rand(self.N_dim, self.N_population)
        self.r_2 = np.random.rand(self.N_dim, self.N_population)

    def ask(self):
        #set random values for r_1 and r_2
        self.r_1 = np.random.rand(self.N_dim, self.N_population)
        self.r_2 = np.random.rand(self.N_dim, self.N_population)
        
        #compute new velocity vector, this is the update for the position of the particle
        self.V = self.w_inertia * self.V + self.c_1 * self.r_1 * (self.p_best - self.X) + \
                 self.c_2 * self.r_2 * (self.g_best - self.X)
                 
        #add the velocity to the position to get the new position
        self.X += self.V

        # Velocity limit this ensures stability so particles don't go crazy
        self.V = np.clip(self.V, -self.Vmax, self.Vmax)
        
         # Check for boundary violations and apply velocity mirroring
         # this flips the velocity of the particle wants to go out of bounds
        exceed_upper = self.X > self.upper_bound
        exceed_lower = self.X < self.lower_bound

        # Reverse velocity for boundary violations
        self.V[exceed_upper] *= -1
        self.V[exceed_lower] *= -1

        # Optional: Reset positions to boundary values useful for hardware weights for example spatial light modulator etc ... 
        self.X[exceed_upper] = self.upper_bound
        self.X[exceed_lower] = self.lower_bound

        return self.X

    def tell(self, reward_table):

        better_reward_idx = np.where(reward_table < self.curr_reward)[0]
        self.curr_reward[better_reward_idx] = reward_table[better_reward_idx]
        self.p_best[:, better_reward_idx] = self.X[:, better_reward_idx]

        best_curr_reward_idx = np.argmin(reward_table)
        best_curr_reward = reward_table[best_curr_reward_idx]

        # Setting the best reward and position
        if best_curr_reward < self.best_reward:
            self.g_best = (self.X[:, best_curr_reward_idx])[:,np.newaxis]
            self.best_reward = best_curr_reward