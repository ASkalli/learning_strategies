import cma
import numpy as np

class CMAES:
    '''wrapper for CMA-ES optimization from the python hansen library'''
    def __init__(self, num_params,mean_init,      # number of model parameters
                    sigma_init=0.10,       # initial standard deviation
                    popsize=255):          # population size

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize

        self.solutions = None


        self.es = cma.CMAEvolutionStrategy( mean_init,
                                            self.sigma_init,
                                            {'popsize': self.popsize})


    def ask(self):
        '''returns the new population of candidates'''
        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = reward_table_result
        self.es.tell(self.solutions, (reward_table).tolist()) 