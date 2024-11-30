import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class ReplayBuffer:
    def __init__(self, max_size, obs_dims, n_agents):
        self.max_size = max_size
        self.obs_dims = obs_dims
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0

        # Initialize buffers for states, actions, rewards, next states, and dones
        self.states = np.zeros((max_size, obs_dims), dtype=np.float32)
        self.actions = np.zeros((max_size, n_agents), dtype=np.int32)
        self.rewards = np.zeros((max_size, n_agents), dtype=np.float32)
        self.next_states = np.zeros((max_size, obs_dims), dtype=np.float32)
        self.dones = np.zeros((max_size, n_agents), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """Store a new experience in the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Update the pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        return self.size

class Q_network(Model):
    def __init__(self, n_actions, n_agents, hidden_sizes=(256, 256)):
        super(Q_network, self).__init__()
        self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
        self.q_value_layer = Dense(n_actions*n_agents, activation=None)
        
    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        q_values = self.q_value_layer(x)  # Output Q-values for each action
        return q_values

class Critic(Model):
    def __init__(self, n_actions, n_agents, hidden_sizes=(256, 256)):
        super(Critic, self).__init__()
        self.critic_1 = Q_network(n_actions=n_actions, n_agents=n_agents, hidden_sizes=hidden_sizes)
        self.critic_2 = Q_network(n_actions=n_actions, n_agents=n_agents, hidden_sizes=hidden_sizes)

    def call(self, x):
        q1 = self.critic_1(x)
        q2 = self.critic_2(x)
        return q1, q2
    
class DDQNAgent:
    def __init__(self, n_agents, obs_dims, n_actions, gamma=0.99, polyak=0.995, lr=3e-4, hidden_sizes=(256,256), verbose=True, name="DDQN"):
        """Agent Initialisation"""
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.polyak = polyak
        self.update_frequency = 100
        self.update_step = 0

        self.critic = Critic(n_actions=n_actions, n_agents=n_agents, hidden_sizes=hidden_sizes)
        self.target_critic = Critic(n_actions=n_actions, n_agents=n_agents, hidden_sizes=hidden_sizes)

        self.q1_optimizers = tf.keras.optimizers.AdamW(learning_rate=lr)
        self.q2_optimizers = tf.keras.optimizers.AdamW(learning_rate=lr)

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
        
    def log_episode(self, stats):
        """Log statistics."""
        with self.logger.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.episodes)
                self.logger.flush()
        self.episodes += 1
    
    def update_network_parameters(self, source_variables, target_variables, tau):
        """Copy weights from critics to target critics"""
        for source_var, target_var in zip(source_variables, target_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)

    def select_action(self, observation, epsilon):
        """Select action based on epsilon-greedy policy."""
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        if np.random.rand() < epsilon:
            # Generate random actions for each agent
            actions = [np.random.randint(self.n_actions) for _ in range(self.n_agents)]
        else:
            q_values = self.critic(observation)
            q_values_reshaped = tf.reshape(q_values[0], (self.n_agents, self.n_actions))
            actions = tf.argmax(q_values_reshaped, axis=1).numpy().flatten()
        return actions

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        self.learn(batch_size, *replay_buffer.sample(batch_size))

    @tf.function
    def learn(self, batch_size, states, actions, rewards, next_states, dones):
        """Train the agent using a batch of experiences from the replay buffer."""

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute target Q-values using the target critic
        target_q1, target_q2 = self.target_critic(next_states)
        target_q_values = tf.minimum(target_q1, target_q2)

        # Reshape target Q-values to (batch_size, self.n_agents, self.n_actions)
        target_q_values_reshaped = tf.reshape(target_q_values, (batch_size, self.n_agents, self.n_actions))

        # Select the maximum Q-value for each agent
        max_target_q_values = tf.reduce_max(target_q_values_reshaped, axis=2)

        # Compute the target for the current Q-values
        targets = rewards + self.gamma * max_target_q_values * (1 - dones)

        with tf.GradientTape(persistent=True) as tape:
            # Compute current Q-values using the critic
            current_q1, current_q2 = self.critic(states)

            # Reshape current Q-values to (batch_size, self.n_agents, self.n_actions)
            current_q1_reshaped = tf.reshape(current_q1, (batch_size, self.n_agents, self.n_actions))
            current_q2_reshaped = tf.reshape(current_q2, (batch_size, self.n_agents, self.n_actions))

            # Use one-hot encoding to gather the Q-values corresponding to the actions taken
            actions_one_hot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)

            q1_values = tf.reduce_sum(current_q1_reshaped * actions_one_hot, axis=2)
            q2_values = tf.reduce_sum(current_q2_reshaped * actions_one_hot, axis=2)

            # Compute the loss
            q1_loss = tf.reduce_mean(tf.square(targets - q1_values))
            q2_loss = tf.reduce_mean(tf.square(targets - q2_values))

        # Compute gradients and update the critic networks
        q1_gradients = tape.gradient(q1_loss, self.critic.critic_1.trainable_variables)
        q2_gradients = tape.gradient(q2_loss, self.critic.critic_2.trainable_variables)

        self.q1_optimizers.apply_gradients(zip(q1_gradients, self.critic.critic_1.trainable_variables))
        self.q2_optimizers.apply_gradients(zip(q2_gradients, self.critic.critic_2.trainable_variables))

        # Update target networks using polyak averaging
        self.update_step+=1
        if self.update_step % self.update_frequency == 0:
            self.update_network_parameters(self.critic.variables, self.target_critic.variables, self.polyak)