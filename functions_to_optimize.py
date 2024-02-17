import numpy as np


# rastrigin function
def f_rastrigin(x):
    N = x.shape[0]
    if N < 2:
        raise ValueError('Dimension should be greater than 2')
    #scale = 10**np.linspace(0, N-1, N)
    scale = 5

    f = 10*np.size(x, 0) + np.sum((scale*x)**2 - 10*np.cos(2*np.pi*scale*x))
    return(f)

# sphere function


def f_sphere(x):

    return(np.sum(x**2))


def f_ssphere(x):

    return(np.sqrt(np.sum(x**2)))
