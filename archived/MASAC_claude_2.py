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
    
# class MultiHeadQ_network(Model):
#     def __init__(self, n_agents, hidden_sizes=(256, 256), activation=tf.nn.relu, use_layer_norm=False):
#         super(MultiHeadQ_network, self).__init__()
#         self.use_layer_norm = use_layer_norm
#         self.n_agents = n_agents
        
#         # Shared layers
#         self.hidden_layers = [Dense(size, activation=None) for size in hidden_sizes]
#         if self.use_layer_norm:
#             self.layer_norms = [LayerNormalization() for _ in hidden_sizes]
            
#         # Separate head for each agent
#         self.q_heads = [Dense(1, activation=None) for _ in range(n_agents)]
#         self.activation = activation

#     def call(self, x, a):
#         x = Concatenate(axis=-1)([x, a])
        
#         # Shared feature extraction
#         for i, layer in enumerate(self.hidden_layers):
#             x = layer(x)
#             if self.use_layer_norm:
#                 x = self.layer_norms[i](x)
#             x = self.activation(x)
            
#         # Get Q-values for each agent
#         q_values = tf.stack([tf.squeeze(head(x), axis=-1) for head in self.q_heads], axis=1)
#         return q_values
    
class MultiHeadQ_network(Model):
    def __init__(self, n_agents, hidden_sizes=(256, 256), activation=tf.nn.relu, use_layer_norm=False):
        super(MultiHeadQ_network, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.n_agents = n_agents
        
        # Shared layers
        self.hidden_layers = [Dense(size, activation=None) for size in hidden_sizes]
        if self.use_layer_norm:
            self.layer_norms = [LayerNormalization() for _ in hidden_sizes]
            
        # Single output layer with n_agents nodes
        self.q_output = Dense(n_agents, activation=None)
        self.activation = activation

    def call(self, x, a):
        x = Concatenate(axis=-1)([x, a])
        
        # Shared feature extraction
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation(x)
            
        # Single dense layer outputting Q-values for all agents
        q_values = self.q_output(x)  # Shape: (batch_size, n_agents)
        return q_values

class MultiHeadCritic(Model):
    def __init__(self, n_agents, hidden_sizes=(256, 256), activation=tf.nn.relu, use_layer_norm=False):
        super(MultiHeadCritic, self).__init__()
        self.critic_1 = MultiHeadQ_network(n_agents, hidden_sizes, activation, use_layer_norm)
        self.critic_2 = MultiHeadQ_network(n_agents, hidden_sizes, activation, use_layer_norm)

    def call(self, x, a):
        q1 = self.critic_1(x, a)  # Shape: [batch_size, n_agents]
        q2 = self.critic_2(x, a)  # Shape: [batch_size, n_agents]
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
        self.critic = MultiHeadCritic(n_agents=n_agents, hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)
        self.target_critic = MultiHeadCritic(n_agents=n_agents, hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)

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
        """Optimized training function using multi-head critic network"""
        # Concatenate observations
        all_obs = tf.concat(batch['obs'], axis=-1)
        all_next_obs = tf.concat(batch['next_obs'], axis=-1)
        
        # Stack actions for easier processing
        all_acts = tf.concat([batch['acts'][:, i:i+1] for i in range(self.n_agents)], axis=1)
        rews = batch['rews']
        dones = batch['dones']
        
        # Get current and next policies for all agents in parallel
        actor_outputs = [actor(obs) for actor, obs in zip(self.actors, batch['obs'])]
        next_actor_outputs = [actor(next_obs) for actor, next_obs in zip(self.actors, batch['next_obs'])]
        
        # Unpack actor outputs
        curr_mus, curr_pis, curr_logps = zip(*actor_outputs)
        _, next_pis, next_logps = zip(*next_actor_outputs)
        
        # Stack policies and log probs
        curr_policies = tf.stack(curr_pis, axis=1)  # [batch_size, n_agents, action_dim]
        next_policies = tf.stack(next_pis, axis=1)
        curr_logprobs = tf.stack(curr_logps, axis=1)  # [batch_size, n_agents]
        next_logprobs = tf.stack(next_logps, axis=1)
        
        # Critic training loop
        for _ in range(self.utd_ratio):
            with tf.GradientTape(persistent=True) as tape:
                # Get Q-values for all agents simultaneously
                # Returns shape: [batch_size, n_agents]
                q1_values, q2_values = self.critic(all_obs, tf.reshape(all_acts, [tf.shape(all_acts)[0], -1]))
                
                # Get next Q-values
                next_actions = tf.reshape(next_policies, [tf.shape(next_policies)[0], -1])
                next_q1, next_q2 = self.target_critic(all_next_obs, next_actions)
                
                # Compute targets for each agent
                next_q_min = tf.minimum(next_q1, next_q2)
                q_targets = tf.stop_gradient(rews[:, tf.newaxis] + 
                                        self.gamma * (1 - dones[:, tf.newaxis]) * 
                                        (next_q_min - self.alpha * next_logprobs))
                
                # Compute critic losses for all agents
                q1_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((q1_values - q_targets) ** 2, axis=1))
                q2_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((q2_values - q_targets) ** 2, axis=1))
                critic_loss = q1_loss + q2_loss
                
            # Update critics
            q1_grad = tape.gradient(q1_loss, self.critic.critic_1.trainable_variables)
            q2_grad = tape.gradient(q2_loss, self.critic.critic_2.trainable_variables)
            self.q1_optimizers.apply_gradients(zip(q1_grad, self.critic.critic_1.trainable_variables))
            self.q2_optimizers.apply_gradients(zip(q2_grad, self.critic.critic_2.trainable_variables))
            
            # Update target networks
            self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=self.polyak)
        
        del tape
        
        # Update alpha
        if self.autotune:
            with tf.GradientTape() as alpha_tape:
                mean_entropy = tf.reduce_mean(curr_logprobs)
                alpha_loss = -tf.exp(self.log_alpha) * (mean_entropy + self.target_entropy)
                
            alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))
        
        """Claude ver"""
        # # Policy updates
        # pi_losses = []
        # for agent_id in range(self.n_agents):
        #     with tf.GradientTape() as pi_tape:
        #         # Get current policy actions and log probs for this agent
        #         _, curr_pi, logp_pi = self.actors[agent_id](batch['obs'][agent_id])
                
        #         # Create action input for critic by replacing this agent's action
        #         policy_actions = curr_policies.copy()
        #         policy_actions = tf.tensor_scatter_nd_update(
        #             policy_actions,
        #             indices=[[i, agent_id] for i in range(tf.shape(policy_actions)[0])],
        #             updates=curr_pi
        #         )
        #         policy_actions_flat = tf.reshape(policy_actions, [tf.shape(policy_actions)[0], -1])
                
        #         # Get Q-values
        #         q1_pi, q2_pi = self.critic(all_obs, policy_actions_flat)
        #         min_q_pi = tf.minimum(q1_pi, q2_pi)[:, agent_id]  # Get this agent's Q-value
                
        #         # Compute policy loss
        #         pi_loss = tf.reduce_mean(self.alpha * logp_pi - min_q_pi)
                
        #     # Update policy
        #     pi_grad = pi_tape.gradient(pi_loss, self.actors[agent_id].trainable_variables)
        #     self.pi_optimizers[agent_id].apply_gradients(zip(pi_grad, self.actors[agent_id].trainable_variables))
        #     pi_losses.append(pi_loss)
        
        # return tf.reduce_mean(pi_losses), critic_loss, self.alpha

        """gpt ver"""
        # Policy updates
        pi_losses = []

        # Construct placeholders for policy actions for all agents
        policy_actions = tf.concat([curr_policies[:, i:i+1] for i in range(self.n_agents)], axis=1)

        # Iterate through agents to update their policies
        for agent_id in range(self.n_agents):
            with tf.GradientTape() as pi_tape:
                # Get current policy actions and log probs for this agent
                _, curr_pi, logp_pi = self.actors[agent_id](batch['obs'][agent_id])
                
                # Replace the specific agent's action with its current policy's action
                updated_policy_actions = tf.concat([
                    policy_actions[:, :agent_id],  # Actions before the current agent
                    curr_pi[:, tf.newaxis],       # Current agent's action
                    policy_actions[:, agent_id+1:]  # Actions after the current agent
                ], axis=1)

                # Flatten for the critic input
                updated_policy_actions_flat = tf.reshape(updated_policy_actions, [tf.shape(policy_actions)[0], -1])

                # Get Q-values
                q1_pi, q2_pi = self.critic(all_obs, updated_policy_actions_flat)
                min_q_pi = tf.minimum(q1_pi, q2_pi)[:, agent_id]  # Extract Q-value for this agent
                
                # Compute policy loss
                pi_loss = tf.reduce_mean(self.alpha * logp_pi - min_q_pi)

            # Update policy
            pi_grad = pi_tape.gradient(pi_loss, self.actors[agent_id].trainable_variables)
            self.pi_optimizers[agent_id].apply_gradients(zip(pi_grad, self.actors[agent_id].trainable_variables))
            pi_losses.append(pi_loss)

        return tf.reduce_mean(pi_losses), critic_loss, self.alpha