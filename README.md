# Learning Strategies

This repository contains implementations of various optimization strategies and their application to toy functions and neural networks. The optimization algorithms, such as **CMA-ES**, **Particle Swarm Optimization (PSO)**, **SPSA**, and others, are encapsulated in Python classes and are used within Jupyter notebooks to demonstrate their effectiveness on different tasks, including training neural networks. The main results from this repository can be found in this paper https://arxiv.org/abs/2503.16943 , in chapters 2,3 and 4. We then applied these algorithms to an physical optical neural network fully implemented in hardware and compared them in terms of performance and convergence efficiency. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Optimization Algorithms](#optimization-algorithms)
- [Neural Network Training](#neural-network-training)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository (or just download it manually):
    ```bash
    git clone https://github.com/ASkalli/learning_strategies.git
    ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running Optimization Algorithms
This repository contains several Python classes for implementing optimization techniques. These classes can be found in the following scripts:
- **PSO**: Implemented in `PSO_obj.py`
- **SPSA**: Implemented in `SPSA_obj.py`
- **CMA-ES**: Implemented in `CMA_obj.py`
- **PEPG**: Implemented in `PEPG_obj.py`
- **Finite Difference Gradient**: Implemented in `Finite_diff_grad.py`

These classes are designed to be used in Jupyter notebooks to optimize either toy functions (like the **Rastrigin** and **Sphere** functions) or to train neural networks. For example, you can find their application in the following notebooks:
- `rastrigin_optimization.ipynb` (for function optimization)
- `sphere_optimization.ipynb` (for function optimization)
- `online_training_MNIST.ipynb`, `online_training_fashion.ipynb` (for neural network training)

### Optimizing Neural Networks
Neural network training is demonstrated using the following datasets:
- **MNIST**: For digit classification.
- **Fashion MNIST**: For fashion image classification.
- **Iris & Wine**: For smaller classification tasks.

These tasks are handled by the notebooks, where optimizers from the Python scripts are used to train the models. I did some hyperparameter scans as well.

## Features

- **Optimizer Classes**: Reusable optimizer classes for PSO, CMA-ES, SPSA, PEPG, and Finite Difference Gradient.
- **Neural Network Training**: Example notebooks demonstrating the use of these optimizers to train neural networks on MNIST, Fashion MNIST, and more.
- **Toy Function Optimization**: Notebooks showcasing how these optimizers work on classic test functions like Rastrigin and Sphere.

## Optimization Algorithms

The following optimization algorithms are implemented and available as Python classes:
- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: A powerful black-box optimization method, works well but is a complete pain if dimensions are too high (covariance matrix scales quadratically with problem dimension) ... 
- **PSO (Particle Swarm Optimization)**: A population-based optimization algorithm, quite fast but hyperparameters can be tricky to optimize, hard to develop are nice intuition for them...
- **PEPG (Policy Evolution with Parameter Perturbations)**: An evolutionary optimization method, the michael jordan of algortihms, works like CMA-ES but less complicated and to be honest in my experience it works better in practice especially no quadratic scaling like CMA-ES...
- **SPSA (Simultaneous Perturbation Stochastic Approximation)**: Gradient approximation technique for optimization, great if you want stuff to go fast and still offers reliable performance ! 
- **Finite Difference Gradient**: A method that estimates gradients through small perturbations in each parameter's direction, this one was for the fun of it, it's ok but in practise I wouldn't use it and just use SPSA.

## Neural Network Training

Neural networks are trained using the above optimization techniques in the following notebooks:
- **MNIST**: `online_training_MNIST.ipynb`
- **Fashion MNIST**: `online_training_fashion.ipynb`
- **Iris & Wine**: `online_training_IRIS_Wine.ipynb`

These notebooks showcase how to apply custom optimization techniques to real-world classification tasks.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository, make your changes, and submit a pull request.

1. Fork the project.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

This amazing blogpost : https://blog.otoro.net/2017/10/29/visual-evolution-strategies/ . By D. Ha, very valuable resource. I basically used it to understand the basic concepts, and based some of my code on it.

CMAES:
Here are some really nice tutorials by the inventor of CMAES Nikolaus Hansen, he also provides matlab and python code on his website, I used that as a basis for the python class I included.
- https://arxiv.org/abs/1604.00772
- https://www.youtube.com/watch?v=7VBKLH3oDuw
- 
A series of youtube videos by a youtuber called cabagecat that explains the blog more in detail and is nice for intuition
- https://www.youtube.com/watch?v=5qCAOyNJROg

PSO:

- https://youtu.be/JhgDMAm-imI?si=jbzP-l99-DbXTj8J


PEPG

https://people.idsia.ch/~juergen/nn2010.pdf

https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
