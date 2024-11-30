import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, n_agents, buffer_size, obs_dims, n_actions, batch_size):
        """
        Replay buffer for multiple agents.
        
        Args:
            n_agents (int): Number of agents.
            buffer_size (int): Maximum number of transitions to store.
            obs_dims (list of int): Dimensions of observations for each agent.
            n_actions (int): Number of actions available to each agent.
            batch_size (int): Number of transitions to sample during training.
        """
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_dims = obs_dims
        self.n_actions = n_actions

        # Initialize the buffers for each agent
        self.buffer = {
            "observations": [[] for _ in range(n_agents)],
            "actions": [[] for _ in range(n_agents)],
            "rewards": [[] for _ in range(n_agents)],
            "next_observations": [[] for _ in range(n_agents)],
            "dones": [[] for _ in range(n_agents)],
        }
        self.size = 0  # Tracks the number of transitions in the buffer

    def store_transition(self, observations, actions, rewards, next_observations, dones):
        """
        Store a new transition for all agents.
        
        Args:
            observations (list of np.ndarray): List of current observations for all agents.
            actions (list of int): List of actions taken by all agents.
            rewards (list of float): List of rewards received by all agents.
            next_observations (list of np.ndarray): List of next observations for all agents.
            dones (list of bool): List of done flags for all agents.
        """
        for agent_idx in range(self.n_agents):
            if self.size < self.buffer_size:
                # Append to buffer if not yet full
                self.buffer["observations"][agent_idx].append(observations[agent_idx])
                self.buffer["actions"][agent_idx].append(actions[agent_idx])
                self.buffer["rewards"][agent_idx].append(rewards[agent_idx])
                self.buffer["next_observations"][agent_idx].append(next_observations[agent_idx])
                self.buffer["dones"][agent_idx].append(dones[agent_idx])
            else:
                # Replace oldest transition when buffer is full
                idx = self.size % self.buffer_size
                self.buffer["observations"][agent_idx][idx] = observations[agent_idx]
                self.buffer["actions"][agent_idx][idx] = actions[agent_idx]
                self.buffer["rewards"][agent_idx][idx] = rewards[agent_idx]
                self.buffer["next_observations"][agent_idx][idx] = next_observations[agent_idx]
                self.buffer["dones"][agent_idx][idx] = dones[agent_idx]

        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self):
        """
        Sample a batch of transitions for all agents.
        
        Returns:
            batch (dict): Dictionary containing sampled transitions for each agent.
        """
        # Ensure we have enough transitions to sample a full batch
        max_index = min(self.size, self.buffer_size)
        batch_indices = np.random.choice(max_index, self.batch_size, replace=False)

        batch = {
            "observations": [[] for _ in range(self.n_agents)],
            "actions": [[] for _ in range(self.n_agents)],
            "rewards": [[] for _ in range(self.n_agents)],
            "next_observations": [[] for _ in range(self.n_agents)],
            "dones": [[] for _ in range(self.n_agents)],
        }

        for agent_idx in range(self.n_agents):
            for idx in batch_indices:
                batch["observations"][agent_idx].append(self.buffer["observations"][agent_idx][idx])
                batch["actions"][agent_idx].append(self.buffer["actions"][agent_idx][idx])
                batch["rewards"][agent_idx].append(self.buffer["rewards"][agent_idx][idx])
                batch["next_observations"][agent_idx].append(self.buffer["next_observations"][agent_idx][idx])
                batch["dones"][agent_idx].append(self.buffer["dones"][agent_idx][idx])

            # Convert lists to NumPy arrays for easier processing
            batch["observations"][agent_idx] = np.array(batch["observations"][agent_idx])
            batch["actions"][agent_idx] = np.array(batch["actions"][agent_idx])
            batch["rewards"][agent_idx] = np.array(batch["rewards"][agent_idx])
            batch["next_observations"][agent_idx] = np.array(batch["next_observations"][agent_idx])
            batch["dones"][agent_idx] = np.array(batch["dones"][agent_idx])

        return batch

    def ready(self):
        return self.size>=self.batch_size
    
class Q_network(Model):
    def __init__(self, n_actions, hidden_sizes=(64, 64)):
        super(Q_network, self).__init__()
        self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
        self.q_value_layer = Dense(n_actions, activation=None)
        
    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        q_values = self.q_value_layer(x)  # Output Q-values for each action
        return q_values
    
class DDQNAgent:
    def __init__(self, n_agents, obs_dims, n_actions, gamma=0.99, polyak=0.995, lr=1e-4, hidden_sizes=(64,64), verbose=True, name="DDQN"):
        """Agent Initialisation"""
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.polyak = polyak
        self.update_frequency = 100
        self.update_step = 0
        self.critics = []
        self.target_critics = []
        self.q_optimizers = []
        for i in range(n_agents):
            self.critics.append(Q_network(n_actions=n_actions, hidden_sizes=hidden_sizes))
            self.target_critics.append(Q_network(n_actions=n_actions, hidden_sizes=hidden_sizes))
            self.q_optimizers.append(tf.keras.optimizers.Adam(learning_rate=lr))
            dummy_obs = tf.keras.Input(shape=(obs_dims[i],), dtype=tf.float32)
            self.critics[i](dummy_obs)
            self.target_critics[i](dummy_obs)
            self.update_network_parameters(self.critics[i].variables, self.target_critics[i].variables, tau=1.0)

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

    def select_action(self, observations, epsilon):
        """
        Select actions for all agents based on their local observations using epsilon-greedy policy.
        
        Args:
            observations (list of np.ndarray): List of local observations, one per agent.
            epsilon (float): Probability of choosing a random action (exploration).
        
        Returns:
            actions (list of int): List of actions chosen by each agent.
        """
        actions = []
        for agent_idx, obs in enumerate(observations):
            # Epsilon-greedy decision
            if np.random.rand() < epsilon:
                # Exploration: choose a random action
                action = np.random.randint(0, self.n_actions)
            else:
                # Exploitation: choose the action with the highest Q-value
                obs_tensor = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)  # Add batch dimension
                q_values = self.critics[agent_idx](obs_tensor)  # Get Q-values for the current agent
                action = tf.argmax(q_values, axis=1).numpy()[0]  # Select the action with highest Q-value
            actions.append(action)
        return actions
    
    @tf.function
    def train(self, replay_buffer):
        """
        Train the agents using the DDQN algorithm.
        
        Args:
            replay_buffer (ReplayBuffer): The shared replay buffer storing transitions for all agents.
            tau (float): The soft update coefficient for the target network.
        
        Returns:
            loss_per_agent (list of float): The training loss for each agent.
        """

        # Sample a batch of transitions
        batch = replay_buffer.sample_batch()

        # For each agent, compute loss and update networks
        for agent_idx in range(self.n_agents):
            obs_batch = tf.convert_to_tensor(batch["observations"][agent_idx], dtype=tf.float32)
            actions_batch = tf.convert_to_tensor(batch["actions"][agent_idx], dtype=tf.int32)
            rewards_batch = tf.convert_to_tensor(batch["rewards"][agent_idx], dtype=tf.float32)
            next_obs_batch = tf.convert_to_tensor(batch["next_observations"][agent_idx], dtype=tf.float32)
            dones_batch = tf.convert_to_tensor(batch["dones"][agent_idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                # Compute Q-values for the current state
                q_values = self.critics[agent_idx](obs_batch)  # Shape: (batch_size, n_actions)
                q_values = tf.gather(q_values, actions_batch, batch_dims=1)  # Gather Q-values for taken actions

                # Compute target Q-values using target critic
                target_q_values = self.target_critics[agent_idx](next_obs_batch)
                next_actions = tf.argmax(self.critics[agent_idx](next_obs_batch), axis=1)  # Double Q-learning
                target_q_values = tf.gather(target_q_values, next_actions, batch_dims=1)

                # Compute TD target
                td_target = rewards_batch + self.gamma * target_q_values * (1.0 - dones_batch)

                # Compute the loss
                loss = tf.reduce_mean(tf.square(td_target - q_values))

            # Apply gradients to update the critic network
            gradients = tape.gradient(loss, self.critics[agent_idx].trainable_variables)
            self.q_optimizers[agent_idx].apply_gradients(zip(gradients, self.critics[agent_idx].trainable_variables))

            # Soft update of target network
            self.update_network_parameters(self.critics[agent_idx].variables, 
                                           self.target_critics[agent_idx].variables, 
                                           tau=self.polyak)