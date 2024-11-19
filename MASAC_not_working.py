import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, LayerNormalization

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MASACReplayBuffer:
    def __init__(self, obs_dims, act_dims, size, n_agents, batch_size):
        self.n_agents = n_agents
        
        # Initializing buffers for all agents
        self.obs_buf = [np.zeros([size] + [obs_dim], dtype=np.float32) for obs_dim in obs_dims]
        self.act_buf = np.zeros([size, n_agents] + [act_dims], dtype=np.float32)
        self.rew_buf = np.zeros([size], dtype=np.float32)
        self.next_obs_buf = [np.zeros([size] + [obs_dim], dtype=np.float32) for obs_dim in obs_dims]
        self.done_buf = np.zeros([size], dtype=np.float32)
        
        self.ptr = 0
        self.max_size = size
        self.size = 0
        self.batch_size = batch_size

    def store(self, obs, acts, rews, next_obs, dones):
        """
        Store the experience tuple for all agents at the current time step.
        
        Args:
        - obs: List of observations for each agent at the current time step.
        - acts: List of actions for each agent at the current time step.
        - rews: List of rewards for each agent at the current time step.
        - next_obs: List of next observations for each agent.
        - dones: List of done flags for each agent.
        """
        for i in range(self.n_agents):
            self.obs_buf[i][self.ptr] = obs[i]
            self.next_obs_buf[i][self.ptr] = next_obs[i]
        self.act_buf[self.ptr] = acts
        self.rew_buf[self.ptr] = rews
        self.done_buf[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        """
        Sample a batch of experience tuples from the replay buffer.
        
        Returns a dictionary containing sampled experience for all agents.
        """
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        
        batch = dict(obs=[obs_buf[idxs] for obs_buf in self.obs_buf],
                     acts=self.act_buf[idxs],
                     rews=self.rew_buf[idxs],
                     next_obs=[next_obs_buf[idxs] for next_obs_buf in self.next_obs_buf],
                     dones=self.done_buf[idxs])
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
        # Compute the Gaussian likelihood. Faster than using tfp.
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

class MASACAgent:
    def __init__(self, n_agents, obs_dims, act_dims, action_bound=1, gamma=0.99, polyak=0.995, lr=3e-4, alpha=0.2, autotune=True, critic_hidden_sizes=(256,256), actor_hidden_sizes=(64,64), use_layer_norm=False, utd_ratio=1, verbose=True, name="MASAC"):
        """Agent Initialisation"""
        self.n_agents = n_agents
        self.gamma = gamma
        self.polyak = polyak
        self.autotune = autotune
        self.utd_ratio = utd_ratio

        if autotune:
            self.target_entropy = -tf.reduce_prod(act_dims).numpy()
            self.log_alpha = tf.Variable(0, dtype=tf.float32, trainable=True)
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.alpha = tf.Variable(initial_value=tf.exp(self.log_alpha), trainable=False, dtype=tf.float32)
        else:
            self.alpha = alpha

        # Initialize actors and critics for each agent
        self.actors = [Actor(action_dim=act_dims, hidden_sizes=actor_hidden_sizes, action_scale=action_bound) for _ in range(n_agents)]
        self.critic = Critic(hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)
        self.target_critic = Critic(hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)

        # Optimizers
        self.pi_optimizers = [tf.keras.optimizers.Adam(learning_rate=lr) for _ in range(n_agents)]
        self.q1_optimizers = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q2_optimizers = tf.keras.optimizers.Adam(learning_rate=lr)

        # Initialize networks with dummy inputs
        dummy_obs = tf.keras.Input(shape=(obs_dims,), dtype=tf.float32)
        dummy_act = tf.keras.Input(shape=(n_agents,), dtype=tf.float32)
        self.critic(dummy_obs, dummy_act)
        self.target_critic(dummy_obs, dummy_act)
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
    
    def select_action(self, o, deterministic=False):
        """Sample from policy"""
        actions = []
        for i, obs in enumerate(o):
            obs = tf.convert_to_tensor(obs.reshape(1, -1), dtype=tf.float32)
            if deterministic:
                mu, _, _ = self.actors[i](obs)
                actions.append(mu.numpy()[0])
            else:
                _, pi, _ = self.actors[i](obs)
                actions.append(pi.numpy()[0])
        
        return actions
        
    def update_network_parameters(self, source_variables, target_variables, tau):
        """Copy weights from critics to target critics"""
        for source_var, target_var in zip(source_variables, target_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)
    
    @tf.function
    def train(self, batch):
        """Version 2 of the multi-agent SAC algorithm struggles primarily due to its centralized 
        handling of observations, actions, and log-probabilities, which overlooks agent-specific 
        contributions and dependencies. By concatenating all observations and actions into a single 
        representation, it dilutes the individual agents' roles in the learning process. This design 
        also assumes independence among agents when summing log-probabilities for entropy, which can 
        lead to overestimated entropy and misaligned exploration. Consequently, the global critic 
        and policy updates fail to respect the nuanced interdependencies between agents, resulting 
        in suboptimal coordination.

        Furthermore, the policy loss in Version 2 computes a single value (min_q_pi) for all agents, 
        neglecting their individual Q-value contributions. This uniform treatment undermines 
        agent-specific policy optimization and can cause conflicts in action selection. These 
        design flaws collectively lead to poor learning dynamics, as agents cannot effectively 
        adapt to the environment or each other, particularly in cooperative or interdependent tasks. 
        The simplified structure of Version 2, while computationally efficient, sacrifices the modularity 
        and precision required for robust multi-agent reinforcement learning."""
        # Concatenate all observations and actions for the critics
        all_obs = tf.concat(batch['obs'], axis=-1)
        all_next_obs = tf.concat(batch['next_obs'], axis=-1)
        all_acts = tf.concat([batch['acts'][:, i] for i in range(self.n_agents)], axis=-1)
        rews = batch['rews']
        dones = batch['dones']

        all_pi, all_pi_next, all_logp_pi_next, all_logp_pi = [], [], [], []
        for i in range(self.n_agents):
            _, pi, logp_pi = self.actors[i](batch['obs'][i])
            _, pi_next, logp_pi_next = self.actors[i](batch['next_obs'][i])
            all_pi.append(pi)
            all_pi_next.append(pi_next)
            all_logp_pi_next.append(logp_pi_next)
            all_logp_pi.append(logp_pi)
        all_pi = tf.concat(all_pi, axis=-1)
        all_pi_next = tf.concat(all_pi_next, axis=-1)
        logp_pi_next = tf.reduce_sum(tf.convert_to_tensor(all_logp_pi_next), axis=0)
        logp_pi = tf.reduce_sum(tf.convert_to_tensor(all_logp_pi), axis=0)

        for _ in range(self.utd_ratio):
            with tf.GradientTape(persistent=True) as tape:
                q1, q2 = self.critic(all_obs, all_acts)

                # Bellman backup for Q functions
                q_backup = tf.stop_gradient(rews + self.gamma * (1 - dones) * (tf.minimum(*self.target_critic(all_next_obs, all_pi_next)) - self.alpha * logp_pi_next))
                
                # Q function losses
                q1_loss = tf.reduce_mean((q1 - q_backup) ** 2)
                q2_loss = tf.reduce_mean((q2 - q_backup) ** 2)

            # Compute gradients and update critic network parameters
            q1_gradients = tape.gradient(q1_loss, self.critic.critic_1.trainable_variables)
            self.q1_optimizers.apply_gradients(zip(q1_gradients, self.critic.critic_1.trainable_variables))

            q2_gradients = tape.gradient(q2_loss, self.critic.critic_2.trainable_variables)
            self.q2_optimizers.apply_gradients(zip(q2_gradients, self.critic.critic_2.trainable_variables))

            # Update target networks with polyak averaging
            self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=self.polyak)

        # Free up memory
        del tape

        if self.autotune:
            with tf.GradientTape() as alpha_tape:
                # Compute alpha
                alpha = tf.exp(self.log_alpha)

                # Compute alpha_loss
                alpha_loss = -alpha * (logp_pi + self.target_entropy)
                alpha_loss = tf.reduce_mean(alpha_loss)
            # Compute gradients and apply updates
            alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
            
            # Update alpha value (no .numpy(), keep it as a tensor)
            self.alpha.assign(tf.exp(self.log_alpha))
        
        # After critic updates, compute policy loss and update policy network
        pi_losses = []
        for agent_id in range(self.n_agents):
            with tf.GradientTape() as tape:
                # Get actions and logprobs from actor-critic
                logp_pi = self.actors[agent_id](batch['obs'][agent_id])[2]
                q1_pi, q2_pi = self.critic(all_obs, all_pi)

                # Min Double-Q:
                min_q_pi = tf.minimum(q1_pi, q2_pi)

                # Policy loss
                pi_loss = tf.reduce_mean(self.alpha * logp_pi - min_q_pi)

            # Compute gradients and update actor network parameters
            pi_gradients = tape.gradient(pi_loss, self.actors[agent_id].trainable_variables)
            self.pi_optimizers[agent_id].apply_gradients(zip(pi_gradients, self.actors[agent_id].trainable_variables))

            pi_losses.append(pi_loss)

        return sum(pi_losses)/len(pi_losses), q1_loss + q2_loss, self.alpha