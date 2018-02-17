import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
# import tensorflow.contrib.distributions as distributions


# parent class for all VAE variants
class AbstVAE:
    # def __init__(self, seed, experiment_dir, num_epochs, batch_size, model_scope):
    def __init__(self, seed, model_scope):
        self.seed = seed
        # self.experiment_dir = experiment_dir
        # self.num_epochs = num_epochs
        # self.batch_size = batch_size
        self.model_scope = model_scope
        np.random.seed(self.seed)  # set random seed elsewhere?

    # def encoder(self):

    # def decoder(self):

    # def build_graph(self):

    def _build_model(self):
        raise NotImplementedError

    # def sample(self, z):


class VAE(AbstVAE):
    def __init__(self, x_dims, z_dim=100, hidden_dim=500, lr=.01, seed=123, model_name="vae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims  # TODO: figure out how to deal with channels/color images
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[None, int(np.prod(self.x_dims))], name="X")
        self.noise = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="noise")

        # set up network
        with tf.variable_scope("encoder"):
            # for now, hardcoding model architecture as that specified in paper
            # TODO: allow for variable definition of model architecture

            enet = layers.fully_connected(self.x, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            params = layers.fully_connected(enet, num_outputs=self.z_dim * 2, activation_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            mu = tf.nn.sigmoid(params[:, :self.z_dim])

            # TODO: taken from altosaar's implementation, change this
            sigma = 1e-6 + tf.nn.softplus(params[:, self.z_dim:])  # need to ensure std dev positive

        z = mu + sigma * self.noise

        with tf.variable_scope("decoder"):
            # for now, hardcoding model architecture as that specified in paper
            # TODO: allow for variable definition of model architecture

            dnet = layers.fully_connected(z, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            # any point in making x_hat accessible? ability to sample images once model trained?
            self.x_hat = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)),
                                                activation_fn=tf.nn.sigmoid,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                biases_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                                )  # Bernoulli MLP decoder

        nll_loss = -tf.reduce_sum(self.x * tf.log(1e-8 + self.x_hat) +
                                  (1 - self.x) * tf.log(1e-8 + 1 - self.x_hat), 1)  # Bernoulli nll
        kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)
        self.loss = tf.reduce_mean(nll_loss + kl_loss)
        self.elbo = -1.0 * tf.reduce_mean(nll_loss + kl_loss)

        # in original paper, lr chosen from {0.01, 0.02, 0.1} depending on first few iters training performance
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        # tensorboard summaries
        x_img = tf.reshape(self.x, [-1] + self.x_dims)
        xhat_img = tf.reshape(self.x_hat, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        tf.summary.image('reconstruction', xhat_img)
        tf.summary.scalar('reconstruction_loss', tf.reduce_mean(nll_loss))
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('elbo', self.elbo)
        self.merged = tf.summary.merge_all()
