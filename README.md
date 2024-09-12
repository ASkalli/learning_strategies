Learning Strategies
This repository contains implementations and experiments on various optimization techniques and learning strategies, such as CMA-ES, Particle Swarm Optimization (PSO), and Stochastic Gradient Descent (SGD). It also includes examples of neural network training using datasets like Fashion MNIST, MNIST, Iris, and Wine datasets.

Table of Contents
Installation
Usage
Features
Optimization Techniques
Neural Network Training
Contributing
License
Installation

Usage
Running Optimization Algorithms
The repository contains several scripts and notebooks for running various optimization techniques on different objective functions:

CMA-ES: Check the CMA_obj.py and python_CMA.py scripts.
PEPG: Use PEPG_obj.py for PEPG optimization loops.
PSO: Implemented in PSO_obj.py.
You can test these algorithms on functions like Rastrigin and Sphere, using the corresponding notebooks (rastrigin_optimization.ipynb, sphere_optimization.ipynb).

Neural Network Training
For training neural networks on popular datasets (MNIST, Fashion MNIST, Iris, Wine), you can refer to the following Jupyter Notebooks:

online_training_MNIST.ipynb
online_training_fashion.ipynb
online_training_IRIS_Wine.ipynb
These notebooks showcase different online training strategies with neural networks.

Features
Learning Rate Scans: Optimized learning rate scans for SPSA.
Comparison of Strategies: Comparison of various optimization strategies on common machine learning tasks (e.g., Fashion MNIST).
Custom Objective Functions: Functions like Rastrigin and Sphere for optimization experiments.
Neural Network Utility Functions: Neural network utility scripts (NN_utils.py, NN_utils_IRIS.py).
Optimization Techniques
CMA-ES (Covariance Matrix Adaptation Evolution Strategy): A powerful black-box optimization method.
PSO (Particle Swarm Optimization): A population-based optimization algorithm.
PEPG (Policy Evolution with Parameter Perturbations): An evolutionary optimization method.
SPSA (Simultaneous Perturbation Stochastic Approximation): Gradient approximation technique for optimization.
Neural Network Training
This repository includes examples of training neural networks using online training strategies on various datasets, including:

MNIST: A dataset of handwritten digits.
Fashion MNIST: A dataset of fashion images.
Iris & Wine: Popular datasets for classification tasks.
Contributing
If you'd like to contribute to the project, feel free to fork the repository, make your changes, and submit a pull request.

Fork the project.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.
