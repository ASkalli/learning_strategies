# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:18:06 2024

@author: Admin
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions_to_optimize import f_rastrigin
from CMA_obj import CMA_opt
from es_utils import CMAES, PEPG



# defines a function to use solver to solve fit_func
def test_solver(solver,MAX_ITERATION,fit_func):
  history = []
  for j in range(MAX_ITERATION):
    solutions = solver.ask()
    fitness_list = np.zeros(solver.popsize)
    for i in range(solver.popsize):
      fitness_list[i] = fit_func(solutions[i])
    solver.tell(fitness_list)
    result = solver.result() # first element is the best solution, second element is the best fitness
    history.append(result[1])
    if (j+1) % 100 == 0:
      print("fitness at iteration", (j+1), result[1])
  print("local optimum discovered by solver:\n", result[0])
  print("fitness score at this local optimum:", result[1])
  return history







#test rastrigin function 
lower_bound = -1
upper_bound = 1
x_1 = np.linspace(lower_bound,upper_bound,100)[:,np.newaxis]
x_2 = np.linspace(lower_bound,upper_bound,100)[:,np.newaxis]

X_1, X_2 = np.meshgrid(x_1,x_2)

X = np.concatenate([X_1.ravel()[:,np.newaxis] , X_2.ravel()[:,np.newaxis]],axis=1)
F = np.zeros([X.shape[0],1])

for i in range(X.shape[0]):
    F[i] = f_rastrigin(X[i,:])
F_2 = np.reshape(F,[len(x_1),len(x_2)])

F_2 = np.reshape(np.apply_along_axis(f_rastrigin, 1, X),newshape=[len(x_1),len(x_2)])

plt.contourf(X_1,X_2,F_2)






epochs = 4000
N_dim = 1000
pop_size = 100
init_pos = (upper_bound - lower_bound) * np.random.rand(N_dim, 1) + lower_bound
best_reward = []





# defines PEPG (NES) solver
PEPG_optimizer = PEPG(N_dim,                        # number of model parameters
            #mean= init_pos.squeeze(),
            sigma_init=0.5,                  # initial standard deviation
            learning_rate=0.01,               # learning rate for standard deviation
            learning_rate_decay=1.0,       # don't anneal the learning rate
            popsize=pop_size+1,             # population size
            average_baseline=False,          # set baseline to average of batch
            weight_decay=0.00,            # weight decay coefficient
            rank_fitness=True,           # use rank rather than fitness numbers
            forget_best=False)     


#pepg_history = test_solver(PEPG_optimizer,epochs,f_rastrigin)


for i in range(epochs):
    
    coordinates = PEPG_optimizer.ask()
    coordinates = coordinates.T
    rewards = np.reshape(np.apply_along_axis(f_rastrigin, 1, coordinates.T), newshape=[np.shape(coordinates)[1], 1])
    #rewards = rewards
    PEPG_optimizer.tell(rewards.squeeze())
    
    best_reward.append(np.min(rewards))
    #print best reward after 100 iterations
    if i%100==0:
        print('Best reward at iteration {}: {}'.format(i, np.min(rewards)))
    