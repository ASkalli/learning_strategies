#simple Adam optimizer class
import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = np.array(params)
        self.lr = lr  # Learning rate
        self.beta1 = beta1  # Decay rate for the first moment estimates
        self.beta2 = beta2  # Decay rate for the second moment estimates
        self.epsilon = epsilon  # Small value to prevent division by zero
        self.m = np.zeros_like(params)  # First moment vector
        self.v = np.zeros_like(params)  # Second moment vector
        self.iteration = 0  # Initialization of the timestep

    def step(self, grad):
        """Calculate and return the step to update parameters based on the Adam optimization algorithm."""
        self.iteration += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad  # Update biased first moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)  # Update biased second raw moment estimate

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.iteration)

        # Update parameters
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return step