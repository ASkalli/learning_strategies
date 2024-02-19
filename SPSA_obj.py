"""
Simple SPSA classes for optimization, 

probably too many methods for such a simple algorithm


"""


import numpy as np

class SPSA_opt:
    def __init__(self, params, alpha=0.01,epsilon=1e-5):
        self.params = np.array(params)  # Initial parameters to optimize
        self.alpha = alpha  # Step size
        self.epsilon = epsilon  # Perturbation size
        self.iteration = 0  # Track the current iteration
        self.delta = None

    def perturb_parameters(self):
        """Generate perturbation vector."""
        self.delta = (np.random.rand(*self.params.shape) > 0.5) * 2 - 1  # Random {-1, 1} for each parameter
        params_plus = self.params + self.epsilon * self.delta
        params_minus = self.params - self.epsilon * self.delta
        return params_plus,params_minus

    def approximate_gradient_func(self, loss_func):
        """Approximate the gradient of the loss function; if you have a loss function you can pass as parameter"""
        params_plus = self.params + self.epsilon  * self.delta
        params_minus = self.params - self.epsilon  * self.delta
        loss_plus = loss_func(params_plus)
        loss_minus = loss_func(params_minus)
        gradient = ((loss_plus - loss_minus) / (2 * self.epsilon  * np.var(self.delta))) *self.delta
        return gradient
    
    def approximate_gradient(self,loss_plus ,loss_minus):
        """Approximate the gradient of the loss function, with precalculated loss values"""
        if np.var(self.delta) == 0:
            gradient = (loss_plus - loss_minus) / (2 * self.epsilon) *self.delta
        else :
            gradient = ((loss_plus - loss_minus) / (2 * self.epsilon  * np.var(self.delta))) *self.delta
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