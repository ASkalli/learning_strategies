"""
Simple Finite difference (stupid gradient) class for optimization, 

probably too many methods for such a simple algorithm


"""


import numpy as np

class FD_opt:
    def __init__(self, params,n_perturb, alpha=0.01,epsilon=1e-5):
        self.params = np.array(params)  # Initial parameters to optimize
        self.alpha = alpha  # Step size
        self.epsilon = epsilon  # Perturbation size
        self.iteration = 0  # Track the current iteration
        self.perturb_idx = None  # position of the parameter to perturb
        self.n_perturb = n_perturb   # Number of parameters to perturb

    def generate_perturb_idx(self):
        "Generates a random permutation of indices of the parameters to perturb, these indices are used to build the gradient vector"
        self.perturb_idx = np.random.permutation(self.params.shape[0])[:self.n_perturb]
        
         
    def perturb_parameters(self,index):
        """Perturb parameters and return them: params +/- epsilon."""
              
        params_plus = np.copy(self.params)
        params_minus = np.copy(self.params)
        params_plus[index] += self.epsilon
        params_minus[index] -= self.epsilon
        
        return params_plus, params_minus

    def approximate_gradient_component(self, loss_func, index):
        """Calculate the gradient component for a single parameter index."""
        params_plus, params_minus = self.perturb_parameters(index)
        loss_plus = loss_func(params_plus)
        loss_minus = loss_func(params_minus)
        gradient_component = (loss_plus - loss_minus) / (2 * self.epsilon)
        return gradient_component
    
    def approximate_gradient(self,loss_plus ,loss_minus):
        """Approximate the gradient of the loss function, with precalculated loss values"""
        
        gradient_component = ((loss_plus - loss_minus) / (2 * self.epsilon ))
        return gradient_component
    
    def update_parameters(self, gradient):
        """Update the parameters based on the approximated gradient."""
        #ak = self.alpha / (self.iteration + 1)**0.602
        self.params -= self.alpha * gradient        
        return self.params
    
    def update_parameters_step(self, step):
        """Update the parameters based on the approximated gradient."""
        #ak = self.alpha / (self.iteration + 1)**0.602
        self.params -= step         
        return self.params