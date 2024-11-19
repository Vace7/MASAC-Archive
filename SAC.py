import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, LayerNormalization

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size, batch_size):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.batch_size = batch_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(obs1=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     acts=self.act_buf[idxs],
                     rews=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch

    def __len__(self):
        return self.size
    
    def ready(self):
        return self.size >= self.batch_size
    
class Actor(Model):
    def __init__(self, action_dim, hidden_sizes=(256, 256), action_scale=1, activation=tf.nn.relu, output_activation=None):
        super(Actor, self).__init__()
        self.hidden_layers = [Dense(size, activation=activation) for size in hidden_sizes]
        self.mu_layer = Dense(action_dim, activation=output_activation)
        self.log_std_layer = Dense(action_dim)
        self.action_scale = action_scale

    def gaussian_likelihood(self, x, mu, log_std):
        # Compute the Gaussian likelihood
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS))**2 + 2 * log_std + tf.math.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
    
    def apply_squashing_func(self, mu, pi, logp_pi):
        logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        return mu, pi, logp_pi

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)
        
        return self.apply_squashing_func(mu, pi, logp_pi)

class Q_network(Model):
    def __init__(self, hidden_sizes=(256, 256), activation=tf.nn.relu, use_layer_norm=False):
        super(Q_network, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.hidden_layers = [Dense(size, activation=None) for size in hidden_sizes]
        self.activation = activation
        self.q_value_layer = Dense(1, activation=None)
        if self.use_layer_norm:
            self.layer_norms = [LayerNormalization() for _ in hidden_sizes]

    def call(self, x, a):
        x = Concatenate(axis=-1)([x, a])
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation(x)
        q_value = tf.squeeze(self.q_value_layer(x), axis=-1)
        return q_value

class Critic(Model):
    def __init__(self, hidden_sizes=(256, 256), activation=tf.nn.relu, use_layer_norm=False):
        super(Critic, self).__init__()
        self.critic_1 = Q_network(hidden_sizes=hidden_sizes, activation=activation, use_layer_norm=use_layer_norm)
        self.critic_2 = Q_network(hidden_sizes=hidden_sizes, activation=activation, use_layer_norm=use_layer_norm)

    def call(self, x, a):
        q1 = self.critic_1(x, a)
        q2 = self.critic_2(x, a)
        
        return q1, q2
    
class SACAgent:
    def __init__(self, input_dims, action_dims, action_bound=1, gamma=0.99, polyak=0.995, lr=3e-4, alpha=0.2, autotune=True, critic_hidden_sizes=(256,256), actor_hidden_sizes=(64,64), use_layer_norm=False, utd_ratio=1, verbose=True, name="SAC-multi"):
        """Agent Initialisation"""
        # Hyperparameters
        self.gamma = gamma
        self.polyak = polyak
        self.autotune = autotune
        self.utd_ratio = utd_ratio

        if autotune:
            self.target_entropy = -tf.reduce_prod(action_dims).numpy()
            self.log_alpha = tf.Variable(0, dtype=tf.float32, trainable=True)
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.alpha = tf.Variable(initial_value=tf.exp(self.log_alpha), trainable=False, dtype=tf.float32)
        else:
            self.alpha = alpha

        # Actor-Critic setup
        self.actor = Actor(action_dim=action_dims, hidden_sizes=actor_hidden_sizes, action_scale=action_bound,)
        self.critic = Critic(hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)
        self.target_critic = Critic(hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)

        # Optimizers
        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Initialize networks with dummy inputs
        dummy_obs = tf.keras.Input(shape=(input_dims,), dtype=tf.float32)
        dummy_act = tf.keras.Input(shape=(action_dims,), dtype=tf.float32)
        self.critic(dummy_obs, dummy_act)
        self.target_critic(dummy_obs, dummy_act)
        
        # Initialize target network
        self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=1.0)

        # Logging
        self.episodes = 0
        self.name = name
        if verbose:
            log_dir = f"../logs/{self.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.logger = tf.summary.create_file_writer(log_dir)
        
    def log_episode(self, stats):
        """Log statistics."""
        with self.logger.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.episodes)
                self.logger.flush()
        self.episodes += 1

    def save_models(self):
        """Save actor and critic models."""
        print('... saving models ...')
        self.actor.save(f'model/{self.name}_actor', save_format="tf")
        self.critic.critic_1.save(f'model/{self.name}_critic1', save_format="tf")
        self.critic.critic_2.save(f'model/{self.name}_critic2', save_format="tf")

    def load_models(self):
        """Load actor and critic models."""
        try:
            print('... loading models ...')
            self.actor = tf.keras.models.load_model(f'model/{self.name}_actor')
            self.critic.critic_1 = tf.keras.models.load_model(f'model/{self.name}_critic1')
            self.critic.critic_2 = tf.keras.models.load_model(f'model/{self.name}_critic2')
            self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=1.0)
            print('... Models loaded ...')
        except Exception as e:
            print(e)
            print("Model files failed to load. Using randomly initialized weights.")
    
    def select_action(self, o, deterministic=False):
        """Sample from policy"""
        o = tf.convert_to_tensor(o.reshape(1, -1), dtype=tf.float32)
        if deterministic:
            mu, _, _ = self.actor(o)
            action = mu.numpy()[0]
        else:
            _, pi, _ = self.actor(o)
            action = pi.numpy()[0]
        return action
    
    def update_network_parameters(self, source_variables, target_variables, tau):
        """Copy weights from critics to target critics"""
        for source_var, target_var in zip(source_variables, target_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)
        
    @tf.function
    def train(self, batch):
        """Main learning function"""
        _, _, logp_pi = self.actor(batch['obs1'])
        _, pi_next, logp_pi_next = self.actor(batch['obs2'])

        for _ in range(self.utd_ratio):
            with tf.GradientTape(persistent=True) as tape:
                # Get actions and logprobs from actor-critic
                q1, q2 = self.critic(batch['obs1'], batch['acts'])

                # Bellman backup for Q functions
                q_backup = tf.stop_gradient(batch['rews'] + self.gamma * (1 - batch['done']) * 
                                            (tf.minimum(*self.target_critic(batch['obs2'], pi_next)) - self.alpha * logp_pi_next))
                
                # Q function losses
                q1_loss = tf.reduce_mean((q1 - q_backup) ** 2)
                q2_loss = tf.reduce_mean((q2 - q_backup) ** 2)
                
            # Compute gradients and update critic network parameters
            q1_gradients = tape.gradient(q1_loss, self.critic.critic_1.trainable_variables)
            self.q1_optimizer.apply_gradients(zip(q1_gradients, self.critic.critic_1.trainable_variables))

            q2_gradients = tape.gradient(q2_loss, self.critic.critic_2.trainable_variables)
            self.q2_optimizer.apply_gradients(zip(q2_gradients, self.critic.critic_2.trainable_variables))

            # Update target networks with polyak averaging
            self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=self.polyak)

            del tape

        if self.autotune:
            with tf.GradientTape() as alpha_tape:
                # Compute alpha_loss
                alpha_loss = -tf.exp(self.log_alpha) * (logp_pi + self.target_entropy)
                alpha_loss = tf.reduce_mean(alpha_loss)

            # Compute gradients and apply updates
            alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

        # After critic updates, compute policy loss and update policy network
        with tf.GradientTape() as tape:
            # Get actions and logprobs from actor-critic
            _, pi, logp_pi = self.actor(batch['obs1'])
            q1_pi, q2_pi = self.critic(batch['obs1'], pi)
            
            # Min Double-Q:
            min_q_pi = tf.minimum(q1_pi, q2_pi)

            # Policy loss
            pi_loss = tf.reduce_mean(self.alpha * logp_pi - min_q_pi)

        # Compute gradients and update actor network parameters
        pi_gradients = tape.gradient(pi_loss, self.actor.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_gradients, self.actor.trainable_variables))

        return pi_loss, q1_loss + q2_loss, self.alpha 

    