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
        """Improved training function for continuous multi-agent SAC"""
        """Version 1 excels by treating each agent individually, aligning with decentralized 
        multi-agent reinforcement learning principles. Each agent's observations, actions, and 
        log-probabilities are processed separately, ensuring fine-grained control over their 
        unique contributions. The policy loss and critic updates respect agent-specific Q-values, 
        promoting better alignment between individual agents and their shared environment. 
        Additionally, entropy is averaged across agents, maintaining consistency in exploration 
        while considering the decentralized nature of their policies. This modular approach 
        ensures better coordination and adaptability in tasks with interdependent agents.

        However, Version 1’s reliance on loops for handling agent-specific computations introduces 
        inefficiencies, making it computationally expensive and potentially slower to train. 
        Moreover, the separate handling of agents might lead to over-segmentation, limiting the 
        algorithm's ability to model global interactions among agents effectively. To improve, 
        Version 1 could adopt partial centralization by incorporating shared representations for 
        observations and actions while maintaining agent-specific updates, balancing computational 
        efficiency with decentralized coordination. Additionally, optimizing the handling of agent-specific 
        gradients and reducing redundancy in critic updates could further enhance performance and scalability.
        """
        # Extract batch data
        obs_list = batch['obs']  # List of observations for each agent
        next_obs_list = batch['next_obs']
        acts_list = [batch['acts'][:, i] for i in range(self.n_agents)]  # List of actions for each agent
        rews = batch['rews']
        dones = batch['dones']
        
        # Get current policy distributions and actions for each agent
        curr_policies = []
        curr_actions = []
        curr_logprobs = []
        next_policies = []
        next_actions = []
        next_logprobs = []
        
        for i in range(self.n_agents):
            # Current timestep
            _, pi, logp_pi = self.actors[i](obs_list[i])
            curr_policies.append(pi)
            curr_actions.append(acts_list[i])
            curr_logprobs.append(logp_pi)
            
            # Next timestep
            _, pi_next, logp_pi_next = self.actors[i](next_obs_list[i])
            next_policies.append(pi_next)
            next_actions.append(pi_next)  # Using sampled actions for next state
            next_logprobs.append(logp_pi_next)

        # Critic update loop
        for _ in range(self.utd_ratio):
            with tf.GradientTape(persistent=True) as tape:
                # Current Q-values for actual actions
                q1_list, q2_list = [], []
                for i in range(self.n_agents):
                    # Prepare input for critic by concatenating all observations and keeping current agent's action
                    other_actions = [acts_list[j] for j in range(self.n_agents) if j != i]
                    critic_acts = tf.concat([*other_actions[:i], curr_actions[i], *other_actions[i:]], axis=-1)
                    critic_obs = tf.concat([obs_list[j] for j in range(self.n_agents)], axis=-1)
                    
                    q1, q2 = self.critic(critic_obs, critic_acts)
                    q1_list.append(q1)
                    q2_list.append(q2)
                
                # Next Q-values for policy actions
                next_q1_list, next_q2_list = [], []
                for i in range(self.n_agents):
                    # Similar preparation for next state
                    other_next_actions = [next_actions[j] for j in range(self.n_agents) if j != i]
                    critic_next_acts = tf.concat([*other_next_actions[:i], next_actions[i], *other_next_actions[i:]], axis=-1)
                    critic_next_obs = tf.concat([next_obs_list[j] for j in range(self.n_agents)], axis=-1)
                    
                    next_q1, next_q2 = self.target_critic(critic_next_obs, critic_next_acts)
                    next_q1_list.append(next_q1)
                    next_q2_list.append(next_q2)
                
                # Average Q-values across agents
                q1_value = tf.reduce_mean(tf.stack(q1_list, axis=0), axis=0)
                q2_value = tf.reduce_mean(tf.stack(q2_list, axis=0), axis=0)
                next_q1_value = tf.reduce_mean(tf.stack(next_q1_list, axis=0), axis=0)
                next_q2_value = tf.reduce_mean(tf.stack(next_q2_list, axis=0), axis=0)
                
                # Entropy term for each agent
                total_entropy = tf.reduce_mean(tf.stack([logp for logp in next_logprobs], axis=0), axis=0)
                
                # Compute targets with entropy regularization
                min_next_q = tf.minimum(next_q1_value, next_q2_value)
                q_target = tf.stop_gradient(rews + self.gamma * (1 - dones) * (min_next_q - self.alpha * total_entropy))
                
                # Critic losses
                q1_loss = 0.5 * tf.reduce_mean((q1_value - q_target) ** 2)
                q2_loss = 0.5 * tf.reduce_mean((q2_value - q_target) ** 2)
                critic_loss = q1_loss + q2_loss

            # Update critics
            q1_grad = tape.gradient(q1_loss, self.critic.critic_1.trainable_variables)
            q2_grad = tape.gradient(q2_loss, self.critic.critic_2.trainable_variables)
            self.q1_optimizers.apply_gradients(zip(q1_grad, self.critic.critic_1.trainable_variables))
            self.q2_optimizers.apply_gradients(zip(q2_grad, self.critic.critic_2.trainable_variables))

            # Update target networks
            self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=self.polyak)
            
            del tape

        # Alpha update (if auto-tuning)
        if self.autotune:
            with tf.GradientTape() as alpha_tape:
                # Compute average entropy across all agents
                mean_entropy = tf.reduce_mean(tf.stack([tf.reduce_mean(logp) for logp in curr_logprobs]))
                alpha_loss = -tf.exp(self.log_alpha) * (mean_entropy + self.target_entropy)
                
            alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

        # Policy updates
        pi_losses = []
        for agent_id in range(self.n_agents):
            with tf.GradientTape() as pi_tape:
                # Get current policy actions and log probs
                _, curr_pi, logp_pi = self.actors[agent_id](obs_list[agent_id])
                
                # Prepare actions for critic evaluation
                other_actions = [curr_policies[j] for j in range(self.n_agents) if j != agent_id]
                critic_acts = tf.concat([*other_actions[:agent_id], curr_pi, *other_actions[agent_id:]], axis=-1)
                critic_obs = tf.concat([obs_list[j] for j in range(self.n_agents)], axis=-1)
                
                # Get Q-values for policy actions
                q1_pi, q2_pi = self.critic(critic_obs, critic_acts)
                min_q_pi = tf.minimum(q1_pi, q2_pi)
                
                # Compute policy loss with entropy regularization
                pi_loss = tf.reduce_mean(self.alpha * logp_pi - min_q_pi)
                
            # Update policy
            pi_grad = pi_tape.gradient(pi_loss, self.actors[agent_id].trainable_variables)
            self.pi_optimizers[agent_id].apply_gradients(zip(pi_grad, self.actors[agent_id].trainable_variables))
            pi_losses.append(pi_loss)

        return tf.reduce_mean(pi_losses), critic_loss, self.alpha