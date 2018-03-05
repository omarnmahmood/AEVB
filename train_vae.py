import os
import argparse
import numpy as np

from torch.autograd import Variable

import VAE_NN
from torchvision import datasets, transforms

import time

from torch.optim import Adam, Adagrad, SGD

# TODO: Make this work with the dataloader that Prof. Shen created

#from load_data import load_data


def parse_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument('--model', type=str, default='vae', choices=['vae'],
    #                    help='type of variational autoencoder model (default: vae)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST','Frey', 'cifar'],
                        help='dataset on which to train (default: mnist)\n' +
                             'options: [MNIST,Frey,cifar]')

    # TODO: input checks
    # TODO: add ability to pass hyperparameter values as a .json file
    # ISSUE: any way to add ability to specify encoder/decoder architectures?
    # parser.add_argument('--hparams_file', type=str, default='./hparams.json',
    #                     help='JSON file specifying the hyperparameters for training and record keeping')
    parser.add_argument('--experiment_dir', type=str, default='./experiment',
                        help='directory to which to output training summary and checkpoint files')

    #parser.add_argument('--seed', type=int, default=123, help='seed for rng (default: 123)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='number of samples per batch (default: 100)')
    parser.add_argument('--init_weights', type=bool, default=False,
                        help='initialise weights to N(0,0.01)')
    #parser.add_argument('--checkpoint_freq', type=int, default=100,
    #                    help='frequency (in epochs) with which we save model checkpoints (default: 100)')

    # also allow specification of optimizer to use?
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--z_dim', type=int, default=20, help='dimensionality of latent variable')
    parser.add_argument('--pca_dim', type=int, default=500, help='number of principal components')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # output directories
    # anything else that should be outputted/recorded? logs? example images?
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
    # remove model saving for now
    #checkpoint_path = os.path.join(checkpoint_dir, "model")
    summary_dir = os.path.join(args.experiment_dir, "summaries")
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    #if not os.path.exists(checkpoint_dir):
    #    os.makedirs(checkpoint_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    vae_n = VAE_NN.VAE_Net(args.z_dim,args.dataset, pca_dim=args.pca_dim)

    # make it trainable on the GPU
    vae_n.cuda()
    
    if args.init_weights:
        vae_n.apply(VAE_NN.init_weights)

    #optimizer = Adam(vae_n.parameters(),lr=args.lr, weight_decay = 1.0)

    optimizer = Adam(vae_n.parameters(),lr=args.lr)

    train_data,_ = VAE_NN.get_data_loaders(b_size=args.batch_size,data=args.dataset)

    t = time.time()

    VAE_NN.train(vae_n,optimizer,train_data, VAE_NN.elbo_loss, epochs = args.num_epochs, summary = summary_dir, pca_dim=args.pca_dim)
    t_e = time.time() - t
    print('Seconds for %d epcohs: %d' % (args.num_epochs,t_e))
