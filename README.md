# Liquid-like-deep-learning

Code for paper [Data-driven effective model shows a liquid-like deep learning]([[2007.08093\] Data-driven effective model shows a liquid-like deep learning (arxiv.org)](https://arxiv.org/abs/2007.08093))(arXiv:2007.08093). Here, we propose a statistical mechanics framework by directly building a least structured model of the
high-dimensional weight space, considering realistic structured data, stochastic gradient descent
algorithms, and the computational depth of the network parametrized by weight parameters.



# Requirements

Python 3.8.5



# Acknowledgement

- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)



# Some Instructions

- The model folder covers the most important models in this article, including neural networks (continuous and discrete), inverse Ising problem, and entropy landscape analysis;
- The data folder contains some important data in this article, including the reduced dimensionality version of the MNIST data set, the moments of different sampled data, and the Ising model parameters obtained by inverse Ising algorithms;
- The task folder covers the implementation code of all the experiments in the paper and the data closely related to this task, besides that already in the data folder, where,
  - You need to modify the data (model) path to use these codes;
  - In the task of Pseudo-likelihood fitting, you need to download the software package in https://web.stanford.edu/~boyd/l1_logreg/ to realize the task with L1 regularization;
  - The sampled data is too large to upload, please contact me if you need it;
- Please contact me if you have any questions about this code. My email: zouwx5@mail2.sysu.edu.cn



# Citation

This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.