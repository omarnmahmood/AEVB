import os
import argparse
import numpy as np

from torch.autograd import Variable

import VAE_NN
from torchvision import datasets, transforms

import time

from torch.optim import Adam, Adagrad, SGD
from torch.optim.lr_scheduler import MultiStepLR

def parse_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument('--model', type=str, default='vae', choices=['vae'],
    #                    help='type of variational autoencoder model (default: vae)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST','Frey'],
                        help='dataset on which to train (default: mnist)\n' +
                             'options: [MNIST,Frey]')

    # TODO: input checks
    # TODO: add ability to pass hyperparameter values as a .json file
    # ISSUE: any way to add ability to specify encoder/decoder architectures?
    # parser.add_argument('--hparams_file', type=str, default='./hparams.json',
    #                     help='JSON file specifying the hyperparameters for training and record keeping')
    parser.add_argument('--experiment_dir', type=str, default='./runs',
                        help='directory to which to output training summary and checkpoint files')

    #parser.add_argument('--seed', type=int, default=123, help='seed for rng (default: 123)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of samples per batch (default: 100)')
    parser.add_argument('--init_weights', type=bool, default=False,
                        help='initialise weights to N(0,0.01)')
    parser.add_argument('--test', type=bool, default=False,
                        help='also benchmark against test')
    parser.add_argument('--conditional', type=bool, default=False,
                        help='use label data (if available)')
    parser.add_argument('--IWAE_mode', type=bool, default=True,
                        help='use IWAE paper settings (ignores all other settings)')
    #parser.add_argument('--checkpoint_freq', type=int, default=100,
    #                    help='frequency (in epochs) with which we save model checkpoints (default: 100)')
    parser.add_argument('--optimiser', type=str, default='Adam',
                        help = 'choose the optimiser')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--z_dim', type=int, default=20, help='dimensionality of latent variable')

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

    vae_n = VAE_NN.VAE_Net(args.z_dim,args.dataset,args.conditional)

    # make it trainable on the GPU
    vae_n.cuda()
    
    if args.init_weights:
        vae_n.apply(VAE_NN.init_weights)

    if args.optimiser == 'Adam':
        optimizer = Adam(vae_n.parameters(),lr=args.lr)
    elif args.optimiser == 'Adagrad':
        optimizer = Adagrad(vae_n.parameters(),lr=args.lr)

    batch_size = args.batch_size  
    dataset = args.dataset  

    lr_scheduler = None

    if args.IWAE_mode:
        batch_size = 20
        dataset = 'MNIST'
        optimizer = Adam(vae_n.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-4)
        lr_scheduler = MultiStepLR(optimizer, milestones=[3,9,27,81,243,729,2187], gamma=10**(-1/7))
        vae_n.apply(VAE_NN.init_weights_xavier)

    train_data,test_data = VAE_NN.get_data_loaders(b_size=batch_size,data=dataset)

    if not args.test and not args.IWAE_mode:
        test_data = None

    t = time.time()

    VAE_NN.train(vae_n,optimizer,train_data, VAE_NN.elbo_loss, epochs = args.num_epochs, summary = summary_dir, test_loader = test_data, scheduler = lr_scheduler)
    t_e = time.time() - t
    print('Seconds for %d epcohs: %d' % (args.num_epochs,t_e))
