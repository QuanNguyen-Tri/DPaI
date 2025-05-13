# DPaI: Differentiable Pruning at Initialization with Node-Path Balance Principle

# Requirements

Tested on python==3.10.14, pytorch==2.4.0, cuda==11.8, torchvison==0.19.0, numpy==1.26.4

# Hyperparameters

* alpha: Trade-off between the number of effective nodes/kernels and effective paths. A higher alpha results in a greater number of effective nodes/kernels (0.0 <= alpha <=1.0).
* beta: Trade-off between the number of effective kernels and effective nodes. A higher beta results in a greater number of effective kernels (0.0 <= beta <=1.0).
* lr_score: The learning rate for the score parameters.
* num_steps: The number of steps for updating the score parameters

# Run Experiments

bash experiments/run.sh