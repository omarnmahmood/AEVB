import os
import argparse
import numpy as np
import tensorflow as tf

from vae import VAE
from load_data import load_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vae', choices=['vae'],
                        help='type of variational autoencoder model (default: vae)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'frey_face', 'fashion_mnist'],
                        help='dataset on which to train (default: mnist)\n' +
                             'options: [mnist, frey_face, fashion_mnist, cifar10, cifar100]')

    # TODO: input checks
    # TODO: add ability to pass hyperparameter values as a .json file
    # ISSUE: any way to add ability to specify encoder/decoder architectures?
    # parser.add_argument('--hparams_file', type=str, default='./hparams.json',
    #                     help='JSON file specifying the hyperparameters for training and record keeping')
    parser.add_argument('--experiment_dir', type=str, default='./experiment',
                        help='directory to which to output training summary and checkpoint files')

    parser.add_argument('--seed', type=int, default=123, help='seed for rng (default: 123)')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of samples per batch (default: 128)')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='frequency (in epochs) with which we save model checkpoints (default: 100)')

    # also allow specification of optimizer to use?
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--z_dim', type=int, default=100, help='dimensionality of latent variable')

    return parser.parse_args()


def train():
    return


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    # does the random seed set above also set the random seed for this class instance?
    dataset = load_data(dataset=args.dataset)

    # output directories
    # anything else that should be outputted/recorded? logs? example images?
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    summary_dir = os.path.join(args.experiment_dir, "summaries")
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # TODO: look into session config options
    with tf.Session() as sess:
        # ISSUE: how best to allow for variable specification of the model?
        # does the random seed set above also set the random seed for this class instance?
        model = VAE(x_dims=dataset.train.img_dims, z_dim=args.z_dim, model_name=args.model)

        global_step = 0
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        # initial setup
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_path, global_step=global_step)

        # num_steps = dataset.train.num_examples // args.batch_size
        # for epoch in range(args.num_epochs):
        #     for step in range(num_steps):
        while dataset.train.epochs_completed < args.num_epochs:
            # Dataset class keeps track of steps in current epoch and number epochs elapsed
            batch = dataset.train.next_batch(args.batch_size)
            summary, _ = sess.run(
                [model.merged, model.train_op],
                feed_dict={
                    model.x: batch[0],
                    model.noise: np.random.randn(args.batch_size, args.z_dim)
                })
            global_step += 1

            summary_writer.add_summary(summary, global_step)
            summary_writer.flush()

            if dataset.train.epochs_completed % args.checkpoint_freq == 0:
                saver.save(sess, checkpoint_path, global_step=global_step)

            # TODO: add logging to stdout or a specified log directory
