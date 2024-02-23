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
        "Generates a random perturbation of indices of the parameters to perturb, these indices are used to build the gradient vector"
        self.perturb_idx = np.random.permutation(self.params.shape[0])[:self.n_perturb]
        
         
    def perturb_parameters(self):
        """Perturb parameters and return them: params +/- epsilon."""
        params_plus = self.params + self.epsilon 
        params_minus = self.params - self.epsilon 
        return params_plus,params_minus

    def approximate_gradient_func(self, loss_func):
        """Approximate the gradient of the loss function; if you have a loss function you can pass as parameter"""
        params_plus = self.params + self.epsilon  
        params_minus = self.params - self.epsilon  
        loss_plus = loss_func(params_plus)
        loss_minus = loss_func(params_minus)
        gradient = (loss_plus - loss_minus) / (2 * self.epsilon )
        return gradient
    
    def approximate_gradient(self,loss_plus ,loss_minus):
        """Approximate the gradient of the loss function, with precalculated loss values"""
        
        gradient = ((loss_plus - loss_minus) / (2 * self.epsilon ))
        return gradient
    
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