import numpy as np
import matplotlib.pyplot as plt
from functions_to_optimize import f_rastrigin
from CMA_obj import CMA_opt

# Test Rastrigin function
lower_bound = -1
upper_bound = 1
x_1 = np.linspace(lower_bound, upper_bound, 100)[:, np.newaxis]
x_2 = np.linspace(lower_bound, upper_bound, 100)[:, np.newaxis]

X_1, X_2 = np.meshgrid(x_1, x_2)
X = np.concatenate([X_1.ravel()[:, np.newaxis], X_2.ravel()[:, np.newaxis]], axis=1)

F_2 = np.reshape(np.apply_along_axis(f_rastrigin, 1, X), newshape=[len(x_1), len(x_2)])

epochs = 4000
N_dim = 100
pop_size = 100
init_pos = (upper_bound - lower_bound) * np.random.rand(N_dim, 1) + lower_bound
#initialize best_rewards to empty vector of zeros 
best_reward = np.zeros(shape = [epochs,1])

CMA_optimizer = CMA_opt(N_dim, pop_size, select_pop=int(pop_size/2), sigma_init=0.5, mean_init=init_pos)

for i in range(int(epochs)):
    
    coordinates = CMA_optimizer.ask()
    
    rewards = np.reshape(np.apply_along_axis(f_rastrigin, 1, coordinates.T), newshape=[np.shape(coordinates)[1], 1])
    
    CMA_optimizer.tell(reward_table=rewards)
    
    best_reward[i]=(np.min(rewards))
    if i%100==0:
        print('Best reward at iteration {}: {}'.format(i, np.min(rewards)))
        

# Plotting after all iterations
plt.loglog(best_reward)
plt.xlabel('Iteration')
plt.ylabel('Best Reward')
plt.title('CMA-ES Optimization on Rastrigin Function')
plt.grid(True)
plt.show()
