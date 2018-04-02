import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as dbns


# parent class for all VAE variants
class AbstVAE:
    def __init__(self, seed, model_scope):
        self.seed = seed
        self.model_scope = model_scope
        np.random.seed(self.seed)

    def encoder(self, x, reuse, trainable):
        raise NotImplementedError

    def decoder(self, z, reuse, trainable):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError


class BernoulliVAE(AbstVAE):
    def __init__(self, x_dims, z_dim=100, lr=.02, seed=123, hidden_dim=500, model_name="vae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def encoder(self, x, scope="encoder", reuse=False, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            # for now, hardcoding model architecture as that specified in paper
            enet = layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          trainable=trainable)
            z_params = layers.fully_connected(enet, num_outputs=self.z_dim * 2, activation_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              biases_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              trainable=trainable)
        return z_params

    def decoder(self, z, scope="decoder", reuse=False, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            # for now, hardcoding model architecture as that specified in paper
            dnet = layers.fully_connected(z, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01))
            x_hat = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)), activation_fn=tf.nn.sigmoid,
                                           weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           biases_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           trainable=trainable)  # Bernoulli MLP decoder
        return x_hat

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[None, int(np.prod(self.x_dims))], name="X")
        self.noise = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="noise")
        self.p_z = dbns.Normal(loc=tf.zeros_like(self.noise), scale=tf.ones_like(self.noise))

        # encoder
        z_params = self.encoder(self.x)
        z_mu = z_params[:, self.z_dim:]
        z_sigma = tf.exp(z_params[:, :self.z_dim])
        self.q_z = dbns.Normal(loc=z_mu, scale=z_sigma)

        # reparameterization trick
        z = z_mu + tf.multiply(z_sigma, self.p_z.sample())
        # z = self.q_z.sample()

        # decoder
        self.x_hat = self.decoder(z)
        self.p_x_z = dbns.Bernoulli(logits=self.x_hat)

        nll_loss = -tf.reduce_sum(self.x * tf.log(1e-8 + self.x_hat) +
                                  (1 - self.x) * tf.log(1e-8 + 1 - self.x_hat), 1)  # Bernoulli nll
        kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.log(1e-8 + tf.square(z_sigma)) - 1, 1)
        # kl_loss = tf.reduce_sum(dbns.kl_divergence(self.q_z, self.p_z), 1)
        self.loss = tf.reduce_mean(nll_loss + kl_loss)
        self.elbo = -1.0 * tf.reduce_mean(nll_loss + kl_loss)

        # in original paper, lr chosen from {0.01, 0.02, 0.1} depending on first few iters training performance
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        # for sampling
        self.z = self.encoder(self.x, trainable=False, reuse=True)
        self.z_pl = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.sample = self.decoder(self.z_pl, trainable=False, reuse=True)

        # tensorboard summaries
        x_img = tf.reshape(self.x, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        xhat_img = tf.reshape(self.x_hat, [-1] + self.x_dims)
        tf.summary.image('reconstruction', xhat_img)
        tf.summary.scalar('reconstruction_loss', tf.reduce_mean(nll_loss))
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('elbo', self.elbo)
        self.merged = tf.summary.merge_all()


class GaussianVAE(AbstVAE):
    def __init__(self, x_dims, z_dim=100, hidden_dim=500, lr=.02, seed=123, model_name="vae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def encoder(self, x, scope="encoder", reuse=False, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            # for now, hardcoding model architecture as that specified in paper
            enet = layers.fully_connected(x, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          trainable=trainable)
            z_params = layers.fully_connected(enet, num_outputs=self.z_dim * 2, activation_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              biases_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              trainable=trainable)
        return z_params

    def decoder(self, z, scope="decoder", reuse=False, trainable=True):
        with tf.variable_scope("decoder", reuse=reuse):
            # for now, hardcoding model architecture as that specified in paper
            dnet = layers.fully_connected(z, num_outputs=self.hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          trainable=trainable)
            out_params = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)*2), activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                biases_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                trainable=trainable)
        return out_params

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[None, int(np.prod(self.x_dims))], name="X")
        self.noise = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="noise")
        self.p_z = dbns.Normal(loc=tf.zeros_like(self.noise), scale=tf.ones_like(self.noise))

        # encoder
        z_params = self.encoder(self.x)
        z_mu = z_params[:, self.z_dim:]
        z_sigma = tf.exp(z_params[:, :self.z_dim])
        self.q_z = dbns.Normal(loc=z_mu, scale=z_sigma)

        # reparameterization trick
        z = z_mu + tf.multiply(z_sigma, self.p_z.sample())
        # z = self.q_z.sample()

        # decoder
        out_params = self.decoder(z)
        mu = tf.nn.sigmoid(out_params[:, int(np.prod(self.x_dims)):])  # out_mu constrained to (0,1)
        sigma = tf.exp(out_params[:, :int(np.prod(self.x_dims))])
        self.x_hat = mu
        self.p_x_z = dbns.Normal(loc=mu, scale=sigma)

        nll_loss = -tf.reduce_sum(self.p_x_z.log_prob(self.x), 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma) - tf.log(1e-8 + tf.square(z_sigma)) - 1, 1)
        # kl_loss = tf.reduce_sum(dbns.kl_divergence(self.q_z, self.p_z), 1)
        self.loss = tf.reduce_mean(nll_loss + kl_loss)
        self.elbo = -1.0 * tf.reduce_mean(nll_loss + kl_loss)

        # in original paper, lr chosen from {0.01, 0.02, 0.1} depending on first few iters training performance
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        # for sampling
        self.z = self.encoder(self.x, trainable=False, reuse=True)
        self.z_pl = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.sample = self.decoder(self.z_pl, trainable=False, reuse=True)

        # tensorboard summaries
        x_img = tf.reshape(self.x, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        xhat_img = tf.reshape(self.x_hat, [-1] + self.x_dims)
        tf.summary.image('reconstruction', xhat_img)
        tf.summary.scalar('reconstruction_loss', tf.reduce_mean(nll_loss))
        tf.summary.scalar('kl_loss', tf.reduce_mean(kl_loss))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('elbo', self.elbo)
        self.merged = tf.summary.merge_all()


class BernoulliIWAE(AbstVAE):
    def __init__(self, x_dims, z_dim=50, hidden_dim=200, seed=123, batch_size=20, n_samples=5, model_name="iwae"):
        super().__init__(seed=seed, model_scope=model_name)
        self.x_dims = x_dims
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_samples = n_samples
        with tf.variable_scope(self.model_scope):
            self._build_model()

    def encoder(self, x, scope="encoder", hidden_dim=200, z_dim=50, reuse=False, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            # Initialization via heuristic specified by Glorot & Bengio 2010
            enet = layers.fully_connected(x, num_outputs=hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer(),
                                          trainable=trainable)
            enet = layers.fully_connected(enet, num_outputs=hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer(),
                                          trainable=trainable)
            z_params = layers.fully_connected(enet, num_outputs=z_dim * 2, activation_fn=None,
                                              weights_initializer=layers.xavier_initializer(),
                                              biases_initializer=layers.xavier_initializer(),
                                              trainable=trainable)
        return z_params

    def decoder(self, z, scope="decoder", hidden_dim=200, reuse=False, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            # Initialization via heuristic specified by Glorot & Bengio 2010
            dnet = layers.fully_connected(z, num_outputs=hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer(),
                                          trainable=trainable)
            dnet = layers.fully_connected(dnet, num_outputs=hidden_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=layers.xavier_initializer(),
                                          trainable=trainable)
            x_hat = layers.fully_connected(dnet, num_outputs=int(np.prod(self.x_dims)), activation_fn=tf.nn.sigmoid,
                                           weights_initializer=layers.xavier_initializer(),
                                           biases_initializer=layers.xavier_initializer(),
                                           trainable=trainable)  # Bernoulli MLP decoder
        return x_hat

    def _build_model(self):
        # input points
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, int(np.prod(self.x_dims))], name="X")
        x = tf.tile(self.x, multiples=[self.n_samples, 1])
        self.lr = tf.placeholder(tf.float32, shape=(), name="lr")

        self.p_z = dbns.Normal(loc=tf.zeros(shape=[self.batch_size * self.n_samples, self.z_dim]),
                               scale=tf.ones(shape=[self.batch_size * self.n_samples, self.z_dim]))
        # self.p_h1 = dbns.Normal(loc=tf.zeros(shape=[self.batch_size * self.n_samples, 100]),
        #                         scale=tf.ones(shape=[self.batch_size * self.n_samples, 100]))
        # self.p_h2 = dbns.Normal(loc=tf.zeros(shape=[self.batch_size * self.n_samples, 50]),
        #                         scale=tf.ones(shape=[self.batch_size * self.n_samples, 50]))
        # self.p_h1_ = dbns.Normal(loc=tf.zeros(shape=[self.batch_size * self.n_samples, 100]),
        #                          scale=tf.ones(shape=[self.batch_size * self.n_samples, 100]))

        # encoder
        z_params = self.encoder(x)
        z_mu = z_params[:, self.z_dim:]
        z_sigma = tf.exp(z_params[:, :self.z_dim])
        self.q_z = dbns.Normal(loc=z_mu, scale=z_sigma)
        # params_q_h1_x = self.encoder(x, scope="q_h1_x", hidden_dim=200, z_dim=100)
        # h1_mu = params_q_h1_x[:, 100:]
        # h1_sigma = tf.exp(params_q_h1_x[:, :100])
        # self.q_h1_x = dbns.Normal(loc=h1_mu, scale=h1_sigma)
        # h1 = h1_mu + tf.multiply(h1_sigma, self.p_h1.sample())
        # params_q_h2_h1 = self.encoder(h1, scope="q_h2_h1", hidden_dim=100, z_dim=50)
        # h2_mu = params_q_h2_h1[:, 50:]
        # h2_sigma = tf.exp(params_q_h2_h1[:, :50])
        # self.q_h2_h1 = dbns.Normal(loc=h2_mu, scale=h2_sigma)
        # h2 = h2_mu + tf.multiply(h2_sigma, self.p_h2.sample())

        z = z_mu + tf.multiply(z_sigma, self.p_z.sample())

        # params_p_h1_h2 = self.encoder(h2, scope="p_h1_h2", hidden_dim=100, z_dim=100)
        # h1_mu_ = params_p_h1_h2[:, 100:]
        # h1_sigma_ = tf.exp(params_p_h1_h2[:, :100])
        # self.p_h1_h2 = dbns.Normal(loc=h1_mu_, scale=h1_sigma_)
        # h1_ = h1_mu_ + tf.multiply(h1_sigma_, self.p_h1_.sample())
        # x_hat = self.decoder(h1_, hidden_dim=200)
        # x_hat = self.decoder(h1, hidden_dim=200)

        x_hat = self.decoder(z)
        self.out_dbn = dbns.Bernoulli(logits=x_hat)

        log_lik = tf.reduce_sum(x * tf.log(1e-8 + x_hat) + (1 - x) * tf.log(1e-8 + 1 - x_hat), 1)
        neg_kld = tf.reduce_sum(self.p_z.log_prob(z) - self.q_z.log_prob(z), 1)
        # log_lik = (tf.reduce_sum(x * tf.log(1e-8 + x_hat) + (1 - x) * tf.log(1e-8 + 1 - x_hat), 1) +
        #            tf.reduce_sum(self.p_h1_h2.log_prob(h1), 1))
        # neg_kld = (tf.reduce_sum(self.p_h1_h2.log_prob(h1_) - self.q_h1_x.log_prob(h1), 1) +
        #            tf.reduce_sum(self.p_h1.log_prob(h1) - self.q_h1_x.log_prob(h1), 1) +
        #            tf.reduce_sum(self.p_h2.log_prob(h2) - self.q_h2_h1.log_prob(h2), 1))

        # log_lik = (tf.reduce_sum(x * tf.log(1e-8 + x_hat) + (1 - x) * tf.log(1e-8 + 1 - x_hat), 1) +
        #            tf.reduce_sum(self.p_h1_h2.log_prob(h1), 1) + tf.reduce_sum(self.p_h2.log_prob(h2), 1))
        # neg_kld = tf.reduce_sum(self.q_h1_x.log_prob(h1), 1) + tf.reduce_sum(self.q_h2_h1.log_prob(h2), 1)

        # calculate importance weights using logsumexp and exp-normalize tricks
        log_iws = (tf.reshape(log_lik, [self.batch_size, self.n_samples]) -
                   tf.reshape(neg_kld, [self.batch_size, self.n_samples]))
        max_log_iws = tf.reduce_max(log_iws, axis=1, keepdims=True)
        log_iws -= max_log_iws
        self.elbo = tf.reduce_mean(max_log_iws + tf.log(1e-8 + tf.reduce_mean(
            tf.exp(log_iws), axis=1, keepdims=True)))
        self.loss = -self.elbo

        # compute gradients
        log_norm_const = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(log_iws), 1, keepdims=True), 1e-9, np.inf))
        log_norm_iws = tf.reshape(log_iws - log_norm_const, shape=[-1])
        norm_iws = tf.stop_gradient(tf.exp(log_norm_iws))
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(-tf.reshape(log_iws, [-1]) * norm_iws, trainable_vars)
        grads_and_vars = zip(grads, trainable_vars)

        # for now, hardcoding the Adam optimizer parameters used in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=0.0001)
        optimizer.apply_gradients(grads_and_vars)
        self.train_op = optimizer.minimize(self.loss)

        # for sampling
        self.z = self.encoder(self.x, trainable=False, reuse=True)
        self.z_pl = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.sample = self.decoder(self.z_pl, trainable=False, reuse=True)

        # tensorboard summaries
        x_img = tf.reshape(x, [-1] + self.x_dims)
        tf.summary.image('data', x_img)
        sample_img = tf.reshape(x_hat, [-1] + self.x_dims)
        tf.summary.image('samples', sample_img)
        tf.summary.scalar('log_lik', tf.reduce_mean(log_lik))
        tf.summary.scalar('neg_kld', tf.reduce_mean(neg_kld))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('elbo', self.elbo)
        self.merged = tf.summary.merge_all()


