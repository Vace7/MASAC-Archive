import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LayerNormalization

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

class MASACReplayBuffer:
    def __init__(self, obs_dims, size, n_agents, batch_size):
        self.n_agents = n_agents
        
        # Initializing buffers for all agents
        self.obs_buf = [np.zeros([size] + [obs_dim], dtype=np.float32) for obs_dim in obs_dims]
        self.act_buf = np.zeros([size, n_agents], dtype=np.int32)
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
        self.act_buf[self.ptr] = acts  # Assumes actions are integers
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
    def __init__(self, n_actions, hidden_sizes=(256, 256)):
        super(Actor, self).__init__()
        self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
        self.output_layer = Dense(n_actions, activation='softmax')  # Output a probability distribution
    
    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        pi = self.output_layer(x)
        log_probs = tf.math.log(pi + 1e-8)
        return pi, log_probs
    
class Q_network(Model):
    def __init__(self, n_actions, n_agents, hidden_sizes=(256, 256), activation=tf.nn.relu, use_layer_norm=False):
        super(Q_network, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.hidden_layers = [Dense(size, activation=None) for size in hidden_sizes]
        self.activation = activation
        self.q_value_layer = Dense(n_actions*n_agents, activation=None)
        if self.use_layer_norm:
            self.layer_norms = [LayerNormalization() for _ in hidden_sizes]
        
    def call(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation(x)
        q_values = self.q_value_layer(x)  # Output Q-values for each action
        return q_values

class Critic(Model):
    def __init__(self, n_actions, n_agents, hidden_sizes=(256, 256), use_layer_norm=False):
        super(Critic, self).__init__()
        self.critic_1 = Q_network(n_actions=n_actions, n_agents=n_agents, hidden_sizes=hidden_sizes, use_layer_norm=use_layer_norm)
        self.critic_2 = Q_network(n_actions=n_actions, n_agents=n_agents, hidden_sizes=hidden_sizes, use_layer_norm=use_layer_norm)

    def call(self, x):
        q1 = self.critic_1(x)
        q2 = self.critic_2(x)
        return q1, q2

class MASACAgent:
    def __init__(self, n_agents, obs_dims, n_actions, gamma=0.99, polyak=0.995, lr=3e-4, alpha=0.2, autotune=True, critic_hidden_sizes=(256,256), actor_hidden_sizes=(64,64), utd_ratio=1, use_layer_norm=False, verbose=True, name="MASAC"):
        """Agent Initialisation"""
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.polyak = polyak
        self.autotune = autotune
        self.utd_ratio = utd_ratio

        # if autotune:
        #     self.target_entropy = -tf.reduce_prod(n_actions).numpy()
        #     self.log_alpha = tf.Variable(0, dtype=tf.float32, trainable=True)
        #     self.alpha_optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
        #     self.alpha = tf.Variable(initial_value=tf.exp(self.log_alpha), trainable=False, dtype=tf.float32)
        # else:
        #     self.alpha = alpha

        # ---------------------Claude---------------------
        if autotune:
            # Create separate target entropy and alpha for each agent
            self.target_entropy = [-tf.reduce_prod(n_actions).numpy() for _ in range(n_agents)]
            self.log_alphas = [tf.Variable(0, dtype=tf.float32, trainable=True) for _ in range(n_agents)]
            self.alpha_optimizers = [tf.keras.optimizers.AdamW(learning_rate=lr) for _ in range(n_agents)]
            self.alphas = [tf.Variable(initial_value=tf.exp(log_alpha), trainable=False, dtype=tf.float32) 
                        for log_alpha in self.log_alphas]
        else:
            # If not autotuning, still create separate alphas for each agent
            self.alphas = [alpha for _ in range(n_agents)]
        # ------------------------------------------------

        # Actor-Critic setup
        self.actors = [Actor(n_actions=n_actions, hidden_sizes=actor_hidden_sizes) for _ in range(n_agents)]
        self.critic = Critic(n_actions=n_actions, n_agents=n_agents, hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)
        self.target_critic = Critic(n_actions=n_actions, n_agents=n_agents, hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)

        # Optimizers
        self.pi_optimizers = [tf.keras.optimizers.AdamW(learning_rate=lr) for _ in range(n_agents)]
        self.q1_optimizers = tf.keras.optimizers.AdamW(learning_rate=lr)
        self.q2_optimizers = tf.keras.optimizers.AdamW(learning_rate=lr)

        # Initialize networks with dummy inputs
        dummy_obs = tf.keras.Input(shape=(obs_dims,), dtype=tf.float32)
        self.critic(dummy_obs)
        self.target_critic(dummy_obs)
        self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=1.0)

        # Logging
        self.episodes = 0
        self.name = name
        if verbose:
            log_dir = f"../logs/{self.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.logger = tf.summary.create_file_writer(log_dir)
    
    def reset_weights(self, autotune, n_actions, n_agents, alpha, lr, actor_hidden_sizes, critic_hidden_sizes, obs_dims, use_layer_norm):
        if autotune:
            # Create separate target entropy and alpha for each agent
            self.target_entropy = [-tf.reduce_prod(n_actions).numpy() for _ in range(n_agents)]
            self.log_alphas = [tf.Variable(0, dtype=tf.float32, trainable=True) for _ in range(n_agents)]
            self.alpha_optimizers = [tf.keras.optimizers.AdamW(learning_rate=lr) for _ in range(n_agents)]
            self.alphas = [tf.Variable(initial_value=tf.exp(log_alpha), trainable=False, dtype=tf.float32) 
                        for log_alpha in self.log_alphas]
        else:
            # If not autotuning, still create separate alphas for each agent
            self.alphas = [alpha for _ in range(n_agents)]
        # ------------------------------------------------

        # Actor-Critic setup
        self.actors = [Actor(n_actions=n_actions, hidden_sizes=actor_hidden_sizes) for _ in range(n_agents)]
        self.critic = Critic(n_actions=n_actions, n_agents=n_agents, hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)
        self.target_critic = Critic(n_actions=n_actions, n_agents=n_agents, hidden_sizes=critic_hidden_sizes, use_layer_norm=use_layer_norm)

        # Optimizers
        self.pi_optimizers = [tf.keras.optimizers.AdamW(learning_rate=lr) for _ in range(n_agents)]
        self.q1_optimizers = tf.keras.optimizers.AdamW(learning_rate=lr)
        self.q2_optimizers = tf.keras.optimizers.AdamW(learning_rate=lr)

        # Initialize networks with dummy inputs
        dummy_obs = tf.keras.Input(shape=(obs_dims,), dtype=tf.float32)
        self.critic(dummy_obs)
        self.target_critic(dummy_obs)
        self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=1.0)
        self.episodes = 0

        # Clear TensorFlow's memory
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        import gc
        gc.collect()

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
            obs = tf.convert_to_tensor([obs], dtype=tf.float32)
            pi, _ = self.actors[i](obs)
            if deterministic:
                actions.append(tf.argmax(pi, axis=-1).numpy()[0])
            else:
                actions.append(tf.squeeze(tf.random.categorical(tf.math.log(pi), num_samples=1)).numpy())
        return actions
        
    def update_network_parameters(self, source_variables, target_variables, tau):
        """Copy weights from critics to target critics"""
        for source_var, target_var in zip(source_variables, target_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)

    @tf.function
    def train(self, batch):
        """Main learning function"""
        all_obs = tf.concat(batch['obs'], axis=-1)
        all_next_obs = tf.concat(batch['next_obs'], axis=-1)
        all_acts = tf.concat([tf.one_hot(batch['acts'][:, i], depth=self.n_actions) for i in range(self.n_agents)], axis=-1)
        rews = batch['rews']
        dones = batch['dones']

        # Precompute actor outputs
        def get_actor_outputs():
            logp_pi_list, pi_next_list, logp_pi_next_list = [], [], []
            for i in range(self.n_agents):
                logp_pi = self.actors[i](batch['obs'][i])[1]
                pi_next, logp_pi_next = self.actors[i](batch['next_obs'][i])
                logp_pi_list.append(logp_pi)
                pi_next_list.append(pi_next)
                logp_pi_next_list.append(logp_pi_next)
            return logp_pi_list, pi_next_list, logp_pi_next_list

        # Compute actor outputs once for reuse
        logp_pi_list, pi_next_list, logp_pi_next_list = get_actor_outputs()
        logp_pi = tf.concat(logp_pi_list, axis=-1)
        pi_next = tf.concat(pi_next_list, axis=-1)
        logp_pi_next = tf.concat(logp_pi_next_list, axis=-1)

        for _ in range(self.utd_ratio):
            with tf.GradientTape(persistent=True) as tape:
                # Get Q values from the critic
                q1, q2 = self.critic(all_obs)

                # Bellman backup for Q functions
                target_q1, target_q2 = self.target_critic(all_next_obs)
                target_min_q = tf.minimum(target_q1, target_q2)
                min_qf_next_target = tf.reduce_sum(pi_next * (target_min_q - tf.reduce_mean(self.alphas) * logp_pi_next), axis=1)
                # min_qf_next_target = tf.reduce_sum(pi_next * (target_min_q - self.alpha * logp_pi_next), axis=1)
                q_backup = tf.stop_gradient(rews + self.gamma * (1 - dones) * min_qf_next_target)

                # Calculate Q values for the taken actions
                q1_a_values = tf.reduce_sum(q1 * all_acts, axis=1)
                q2_a_values = tf.reduce_sum(q2 * all_acts, axis=1)

                # Q function losses
                q1_loss = tf.reduce_mean((q1_a_values - q_backup) ** 2)
                q2_loss = tf.reduce_mean((q2_a_values - q_backup) ** 2)

            # Compute gradients and update critic network parameters
            q1_gradients = tape.gradient(q1_loss, self.critic.critic_1.trainable_variables)
            self.q1_optimizers.apply_gradients(zip(q1_gradients, self.critic.critic_1.trainable_variables))

            q2_gradients = tape.gradient(q2_loss, self.critic.critic_2.trainable_variables)
            self.q2_optimizers.apply_gradients(zip(q2_gradients, self.critic.critic_2.trainable_variables))

            self.update_network_parameters(self.critic.variables, self.target_critic.variables, tau=self.polyak)
            
            del tape

        # ---------------------Claude---------------------
        # Update alpha for each agent separately
        if self.autotune:
            alpha_losses = []
            for agent_id in range(self.n_agents):
                with tf.GradientTape() as alpha_tape:
                    # Get this agent's log probs
                    logp_pi = logp_pi_list[agent_id]
                    # Compute alpha loss for this agent
                    alpha_loss = -tf.exp(self.log_alphas[agent_id]) * (
                        tf.reduce_mean(logp_pi) + self.target_entropy[agent_id]
                    )
                    alpha_losses.append(alpha_loss)
                
                # Update this agent's alpha
                alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alphas[agent_id]])
                self.alpha_optimizers[agent_id].apply_gradients(
                    zip(alpha_grads, [self.log_alphas[agent_id]])
                )
                self.alphas[agent_id].assign(tf.exp(self.log_alphas[agent_id]))

        # Update policies using agent-specific alphas
        pi_losses = []
        for agent_id in range(self.n_agents):
            with tf.GradientTape() as tape:
                pi, logp_pi = self.actors[agent_id](batch['obs'][agent_id])
                q1_all_agents, q2_all_agents = self.critic(all_obs)
                q1_all_agents = tf.reshape(q1_all_agents, [tf.shape(q1_all_agents)[0], self.n_agents, -1])
                q2_all_agents = tf.reshape(q2_all_agents, [tf.shape(q2_all_agents)[0], self.n_agents, -1])

                q1_values_curr_agent = q1_all_agents[:, agent_id, :]
                q2_values_curr_agent = q2_all_agents[:, agent_id, :]
                min_q_pi = tf.minimum(q1_values_curr_agent, q2_values_curr_agent)

                # Use agent-specific alpha
                pi_loss = tf.reduce_mean(tf.reduce_sum(pi * (
                    self.alphas[agent_id] * logp_pi - min_q_pi), axis=1))

            pi_gradients = tape.gradient(pi_loss, self.actors[agent_id].trainable_variables)
            self.pi_optimizers[agent_id].apply_gradients(zip(pi_gradients, self.actors[agent_id].trainable_variables))

            pi_losses.append(pi_loss)

        return sum(pi_losses)/len(pi_losses), q1_loss + q2_loss, tf.reduce_mean(self.alphas)
        # ------------------------------------------------

        # # Update alpha
        # if self.autotune:
        #     with tf.GradientTape() as alpha_tape:
        #         alpha_loss = -tf.exp(self.log_alpha) * (tf.reduce_mean(logp_pi) + self.target_entropy)
            
        #     alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
        #     self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        #     self.alpha.assign(tf.exp(self.log_alpha))

        # pi_losses = []
        # for agent_id in range(self.n_agents):
        #     with tf.GradientTape() as tape:
        #         pi, logp_pi = self.actors[agent_id](batch['obs'][agent_id])
        #         q1_all_agents, q2_all_agents = self.critic(all_obs)
        #         q1_all_agents = tf.reshape(q1_all_agents, [tf.shape(q1_all_agents)[0], self.n_agents, -1])
        #         q2_all_agents = tf.reshape(q2_all_agents, [tf.shape(q2_all_agents)[0], self.n_agents, -1])

        #         q1_values_curr_agent = q1_all_agents[:, agent_id, :]
        #         q2_values_curr_agent = q2_all_agents[:, agent_id, :]
        #         min_q_pi = tf.minimum(q1_values_curr_agent, q2_values_curr_agent)

        #         pi_loss = tf.reduce_mean(tf.reduce_sum(pi * (self.alpha * logp_pi - min_q_pi), axis=1))

        #     pi_gradients = tape.gradient(pi_loss, self.actors[agent_id].trainable_variables)
        #     self.pi_optimizers[agent_id].apply_gradients(zip(pi_gradients, self.actors[agent_id].trainable_variables))

        #     pi_losses.append(pi_loss)

        # return sum(pi_losses)/len(pi_losses), q1_loss + q2_loss, self.alpha