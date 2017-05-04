import argparse

parser = argparse.ArgumentParser(description='parameters for training a Restricted Boltzmann Machine (RBM) autoencoder on the MNIST dataset utilizing reparameterized Monte Carlo sampling to approximate marginalization over the Bernoulli hidden state')
parser.add_argument('-h_dim', default=100, type=int, help='dimension of the hidden state')
parser.add_argument('-n_mc', default=3, type=int, help='number of Monte Carlo samples to draw')
parser.add_argument('-lr', default=0.004, type=float, help='learning rate')
parser.add_argument('-epochs', default=200, type=int, help='max number of training epochs')
parser.add_argument('-bs', '--batch_size', default=50, type=int, help='batch size')
parser.add_argument('-reg', default=0.0, type=float, help='amount of weight regularization to use.  Because there are so few parameters this is probably cool at 0')
