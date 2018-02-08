import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as distributions
import tensorflow.contrib.layers as layers

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# TODO: implement this if necessary
# def preprocess_mnist():


# parent class for all VAE variants
class AbstVAE:
    def __init__(self, seed, experiment_dir, num_epochs, batch_size, model_scope):
        self.seed = seed
        self.experiment_dir = experiment_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_scope = model_scope
        np.random.seed(self.seed)

        # output directories
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")
        self.summary_dir = os.path.join(self.experiment_dir, "summaries")
        if not os.path.exists(self.experiment_dir):
            os.makedirs(experiment_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

    def build_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    # don't actually use this function, here for remembering the syntax
    def sample_noise(self, shape):
        # return np.random.randn(shape)
        return tf.random_normal(shape, seed=self.seed)


class VAE(AbstVAE):
    def __init__(self, seed, num_epochs, batch_size, experiment_dir="./experiment/", model_name="vae"):
        super().__init__(seed=seed, experiment_dir=experiment_dir, num_epochs=num_epochs, batch_size=batch_size,
                         model_scope=model_name)
        # TODO: ability to interact with other datasets
        # TODO: better way of interacting with mnist than this
        self.mnist = read_data_sets('MNIST_data', one_hot=True)
        self.num_steps = self.mnist.train.num_examples // self.batch_size

        # self.x_dims = x_dims
        self.x_dims = [28, 28, 1]  # TODO: figure out how to read this from dataset
        self.z_dim = 100  # ISSUE: what should this value be?
        with tf.variable_scope(self.model_scope):
            self._build_model()

    # what should dimensions of Gaussians be?
    def _build_model(self):
        # placeholders
        # ISSUE: mnist data already comes in flattened, check shape for other datasets/mnist modules
        self.x = tf.placeholder(tf.float32, shape=[None, int(np.prod(self.x_dims))], name="X")
        self.noise = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="noise")

        # set up network
        with tf.variable_scope("encoder"):
            # for now, hardcoding model architecture
            # TODO: allow for variable definition of model architecture

            # what activation function did they use?
            # constraining sigma to be a diagonal matrix?
            enet = layers.fully_connected(self.x, num_outputs=500, activation_fn=tf.nn.relu)
            enet = layers.fully_connected(enet, num_outputs=500, activation_fn=tf.nn.relu)
            params = layers.fully_connected(enet, num_outputs=self.z_dim*2, activation_fn=None)
            mu = params[:, :self.z_dim]

            # TODO: taken from altosaar's implementation, change this
            sigma = 1e-6 + tf.nn.softplus(params[:, self.z_dim:])  # need to ensure std dev positive

        z = mu + sigma * self.noise

        with tf.variable_scope("decoder"):
            # for now, hardcoding model architecture
            # TODO: allow for variable definition of model architecture

            dnet = layers.fully_connected(z, num_outputs=500, activation_fn=tf.nn.relu)
            dnet = layers.fully_connected(dnet, num_outputs=500, activation_fn=tf.nn.relu)
            # ISSUE: x_hat appears to be saturating after some number of steps (not creating images anymore)
            self.x_hat = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)),
                                                activation_fn=tf.nn.sigmoid)  # ???
            # self.x_hat = tf.reshape(self.x_hat, [-1] + self.x_dims)

        reconstruction_loss = -tf.reduce_sum(self.x * tf.log(1e-8+self.x_hat) +
                                             (1-self.x) * tf.log(1e-8 + 1 - self.x_hat), 1)  # ???
        kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)
        self.loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        # in original paper, lr chosen from {0.01, 0.02, 0.1} depending on first few iters training performance
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # tensorboard summaries
        x_img = tf.reshape(self.x, [-1] + self.x_dims)
        xhat_img = tf.reshape(self.x_hat, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        tf.summary.image('reconstruction', xhat_img)
        tf.summary.scalar('reconstruction_loss', tf.reduce_mean(reconstruction_loss))
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

    def train(self):
        with tf.Session() as sess:
            # TODO: fix global step
            global_step = tf.Variable(0, trainable=False, name="global_step")

            # initialize tf modules
            # TODO: add ability to load checkpoints
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            self.saver.save(sess, self.checkpoint_path, global_step=global_step)

            for epoch in range(self.num_epochs):
                for step in range(self.num_steps):
                    batch = self.mnist.train.next_batch(self.batch_size)
                    summary, global_step, _ = sess.run(
                        [self.merged, tf.train.get_global_step(), self.train_op],
                        feed_dict={
                            self.x: batch[0],
                            self.noise: np.random.randn(self.batch_size, self.z_dim)
                        })

                    self.summary_writer.add_summary(summary, global_step)
                    self.summary_writer.flush()

                if epoch % 500 == 0:
                    self.saver.save(sess, self.checkpoint_path, global_step=global_step)


# for debugging: in actual implementation, this should be separated
if __name__ == "__main__":
    # TODO: remove necessity of extracting mnist data with every instantiation
    vae = VAE(seed=123,
              num_epochs=10000,
              batch_size=128)
    vae.train()
