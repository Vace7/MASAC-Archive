# import datetime
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dense

# from typing import List, Tuple

# class MADDPGReplayBuffer:
#     def __init__(self, n_agents: int, obs_dims: List[int], action_dims: List[int], 
#                  buffer_size: int = 1_000_000, batch_size: int = 256):
#         """
#         Initialize a replay buffer for MADDPG.
        
#         Args:
#             n_agents: Number of agents
#             obs_dims: List of observation dimensions for each agent
#             action_dims: List of action dimensions for each agent
#             buffer_size: Maximum size of the buffer
#             batch_size: Size of batches to sample
#         """
#         self.n_agents = n_agents
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#         self.pointer = 0
#         self.size = 0
        
#         # Initialize buffers for each agent's observations and actions
#         self.observations = [
#             np.zeros((buffer_size, dim), dtype=np.float32) 
#             for dim in obs_dims
#         ]
#         self.next_observations = [
#             np.zeros((buffer_size, dim), dtype=np.float32) 
#             for dim in obs_dims
#         ]
#         self.actions = [
#             np.zeros((buffer_size, dim), dtype=np.float32) 
#             for dim in action_dims
#         ]
        
#         # Initialize rewards and dones for each agent
#         self.rewards = [
#             np.zeros(buffer_size, dtype=np.float32) 
#             for _ in range(n_agents)
#         ]
#         self.dones = [
#             np.zeros(buffer_size, dtype=np.float32) 
#             for _ in range(n_agents)
#         ]

#     def store(self, obs: List[np.ndarray], actions: List[np.ndarray], 
#               rewards: List[float], next_obs: List[np.ndarray], 
#               dones: List[bool]) -> None:
#         """
#         Store a transition in the buffer.
        
#         Args:
#             obs: List of observations for each agent
#             actions: List of actions for each agent
#             rewards: List of rewards for each agent
#             next_obs: List of next observations for each agent
#             dones: List of done flags for each agent
#         """
#         # Store transition
#         for i in range(self.n_agents):
#             self.observations[i][self.pointer] = obs[i]
#             self.actions[i][self.pointer] = actions[i]
#             self.rewards[i][self.pointer] = rewards[i]
#             self.next_observations[i][self.pointer] = next_obs[i]
#             self.dones[i][self.pointer] = dones[i]
        
#         # Update pointer and size
#         self.pointer = (self.pointer + 1) % self.buffer_size
#         self.size = min(self.size + 1, self.buffer_size)

#     def sample(self) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], 
#                              List[tf.Tensor], List[tf.Tensor]]:
#         """
#         Sample a batch of transitions from the buffer.
        
#         Returns:
#             Tuple containing:
#             - List of observation tensors for each agent
#             - List of action tensors for each agent
#             - List of reward tensors for each agent
#             - List of next observation tensors for each agent
#             - List of done tensors for each agent
#         """
#         # Sample indices
#         indices = np.random.randint(0, self.size, size=self.batch_size)
        
#         # Sample experiences for each agent
#         obs_batch = [
#             tf.convert_to_tensor(self.observations[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         action_batch = [
#             tf.convert_to_tensor(self.actions[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         reward_batch = [
#             tf.convert_to_tensor(self.rewards[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         next_obs_batch = [
#             tf.convert_to_tensor(self.next_observations[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         done_batch = [
#             tf.convert_to_tensor(self.dones[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
        
#         return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

#     def ready(self) -> bool:
#         """
#         Check if enough samples are available for training.
        
#         Returns:
#             True if buffer contains at least one batch of samples
#         """
#         return self.size >= self.batch_size

#     def clear(self) -> None:
#         """Clear the replay buffer"""
#         self.pointer = 0
#         self.size = 0
#         for i in range(self.n_agents):
#             self.observations[i].fill(0)
#             self.actions[i].fill(0)
#             self.rewards[i].fill(0)
#             self.next_observations[i].fill(0)
#             self.dones[i].fill(0)

# class Actor(tf.keras.Model):
#     def __init__(self, action_dim, hidden_sizes=(256, 256)):
#         super(Actor, self).__init__()
#         self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
#         self.output_layer = Dense(action_dim, activation='softmax')
        
#     def call(self, state):
#         x = state
#         for layer in self.hidden_layers:
#             x = layer(x)
#         action_probs = self.output_layer(x)
#         return action_probs
    
#     def sample_action(self, state):
#         action_probs = self.call(state)
#         # Use tf.random.categorical for sampling from probability distribution
#         action = tf.random.categorical(tf.math.log(action_probs), 1)
#         return action, action_probs

# class Critic(tf.keras.Model):
#     def __init__(self, obs_dim, n_actions, n_agents, hidden_sizes=(256, 256)):
#         super(Critic, self).__init__()
#         self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
#         self.output_layer = Dense(1)  # Single Q-value output
        
#     def call(self, inputs):
#         states, actions = inputs
  
#         # Concatenate features
#         x = tf.concat([states, actions], axis=-1)
        
#         for layer in self.hidden_layers:
#             x = layer(x)
#         q_value = self.output_layer(x)
#         return q_value

# class MADDPGAgent:
#     def __init__(self, n_agents, obs_dims, n_actions, gamma=0.99, polyak=0.995, 
#                  lr_actor=3e-4, lr_critic=3e-4, critic_hidden_sizes=(256,256), 
#                  actor_hidden_sizes=(64,64), verbose=True, name="MADDPG"):
#         """Agent Initialization"""
#         self.n_agents = n_agents
#         self.n_actions = n_actions
#         self.obs_dims = obs_dims
#         self.gamma = gamma
#         self.polyak = polyak
        
#         # Initialize networks and optimizers for each agent
#         self.actors = []
#         self.target_actors = []
#         self.critics = []
#         self.target_critics = []
#         self.pi_optimizers = []
#         self.q_optimizers = []

#         self.episodes = 0
#         self.name = name
#         if verbose:
#             log_dir = f"../logs/{self.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
#             self.logger = tf.summary.create_file_writer(log_dir)
        
#         for i in range(n_agents):  # Changed from 'in n_agents' to 'range(n_agents)'
#             # Create actor networks
#             self.actors.append(Actor(n_actions, actor_hidden_sizes))
#             self.target_actors.append(Actor(n_actions, actor_hidden_sizes))
            
#             # Create critic networks
#             self.critics.append(Critic(sum(obs_dims), n_actions*n_agents, n_agents, 
#                                      critic_hidden_sizes))
#             self.target_critics.append(Critic(sum(obs_dims), n_actions*n_agents, n_agents,
#                                             critic_hidden_sizes))
            
#             # Create optimizers
#             self.pi_optimizers.append(tf.keras.optimizers.AdamW(learning_rate=lr_actor))
#             self.q_optimizers.append(tf.keras.optimizers.AdamW(learning_rate=lr_critic))
            
#             # Initialize networks with dummy data
#             self._initialize_networks(i)
    
#     def _initialize_networks(self, agent_idx):
#         """Initialize networks with dummy data"""
#         dummy_state = tf.zeros((1, self.obs_dims[agent_idx]))
#         dummy_full_state = tf.zeros((1, sum(self.obs_dims)))
#         dummy_actions = tf.zeros((1, self.n_actions*self.n_agents))
        
#         # Initialize actor networks
#         self.actors[agent_idx](dummy_state)
#         self.target_actors[agent_idx](dummy_state)
        
#         # Initialize critic networks
#         self.critics[agent_idx]([dummy_full_state, dummy_actions])
#         self.target_critics[agent_idx]([dummy_full_state, dummy_actions])
        
#         # Copy initial weights to targets
#         self.update_target_networks(agent_idx, tau=1.0)
    
#     def update_target_networks(self, agent_idx, tau):
#         """Update target networks using polyak averaging"""
#         # Update actor target
#         for source, target in zip(self.actors[agent_idx].variables, 
#                                 self.target_actors[agent_idx].variables):
#             target.assign(tau * source + (1 - tau) * target)
        
#         # Update critic target
#         for source, target in zip(self.critics[agent_idx].variables, 
#                                 self.target_critics[agent_idx].variables):
#             target.assign(tau * source + (1 - tau) * target)
    
#     @tf.function
#     def train_step(self, states, actions, rewards, next_states, dones):
#         """Training step for all agents"""
#         for i in range(self.n_agents):
#             self._update_critic(i, states, actions, rewards[i], next_states, dones[i])
#             self._update_actor(i, states)
#             self.update_target_networks(i, self.polyak)
    
#     def _update_critic(self, agent_idx, states, actions, rewards, next_states, dones):
#         """Update critic for a specific agent"""
#         with tf.GradientTape() as tape:
#             # Get target actions for next states
#             target_actions = [target_actor(next_state) 
#                             for target_actor, next_state 
#                             in zip(self.target_actors, next_states)]
            
#             # Calculate target Q-value
#             target_q = self.target_critics[agent_idx](
#                 [tf.concat(next_states, axis=1), 
#                  tf.concat(target_actions, axis=1)]
#             )
            
#             target_q = rewards + self.gamma * target_q * (1 - dones)
            
#             # Calculate current Q-value
#             current_q = self.critics[agent_idx](
#                 [tf.concat(states, axis=1), 
#                  tf.concat(actions, axis=1)]
#             )
            
#             # Calculate critic loss
#             critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        
#         # Update critic
#         critic_grads = tape.gradient(critic_loss, self.critics[agent_idx].trainable_variables)
#         self.q_optimizers[agent_idx].apply_gradients(
#             zip(critic_grads, self.critics[agent_idx].trainable_variables)
#         )
        
#         return critic_loss
    
#     def _update_actor(self, agent_idx, states):
#         """Update actor for a specific agent"""
#         with tf.GradientTape() as tape:
#             # Get current actions for all agents
#             current_actions = []
#             for i, (actor, state) in enumerate(zip(self.actors, states)):
#                 if i == agent_idx:
#                     action = actor(state)
#                 else:
#                     action = tf.stop_gradient(actor(state))
#                 current_actions.append(action)
            
#             # Calculate actor loss (negative Q value)
#             actor_loss = -tf.reduce_mean(
#                 self.critics[agent_idx](
#                     [tf.concat(states, axis=1), 
#                      tf.concat(current_actions, axis=1)]
#                 )
#             )
        
#         # Update actor
#         actor_grads = tape.gradient(actor_loss, self.actors[agent_idx].trainable_variables)
#         self.pi_optimizers[agent_idx].apply_gradients(
#             zip(actor_grads, self.actors[agent_idx].trainable_variables)
#         )
        
#         return actor_loss
    
#     def log_episode(self, stats):
#         """Log statistics."""
#         with self.logger.as_default():
#             for key, value in stats.items():
#                 tf.summary.scalar(key, value, step=self.episodes)
#                 self.logger.flush()
#         self.episodes += 1

# import datetime
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Reshape

# from typing import List, Tuple

# # ReplayBuffer remains the same...
# class MADDPGReplayBuffer:
#     def __init__(self, n_agents: int, obs_dims: List[int], action_dims: List[int], 
#                  buffer_size: int = 1_000_000, batch_size: int = 256):
#         self.n_agents = n_agents
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#         self.pointer = 0
#         self.size = 0
        
#         self.observations = [
#             np.zeros((buffer_size, dim), dtype=np.float32) 
#             for dim in obs_dims
#         ]
#         self.next_observations = [
#             np.zeros((buffer_size, dim), dtype=np.float32) 
#             for dim in obs_dims
#         ]
#         self.actions = [
#             np.zeros((buffer_size, dim), dtype=np.float32) 
#             for dim in action_dims
#         ]
#         self.rewards = [
#             np.zeros(buffer_size, dtype=np.float32) 
#             for _ in range(n_agents)
#         ]
#         self.dones = [
#             np.zeros(buffer_size, dtype=np.float32) 
#             for _ in range(n_agents)
#         ]

#     def store(self, obs: List[np.ndarray], actions: List[np.ndarray], 
#               rewards: List[float], next_obs: List[np.ndarray], 
#               dones: List[bool]) -> None:
#         for i in range(self.n_agents):
#             self.observations[i][self.pointer] = obs[i]
#             self.actions[i][self.pointer] = actions[i]
#             self.rewards[i][self.pointer] = rewards[i]
#             self.next_observations[i][self.pointer] = next_obs[i]
#             self.dones[i][self.pointer] = dones[i]
        
#         self.pointer = (self.pointer + 1) % self.buffer_size
#         self.size = min(self.size + 1, self.buffer_size)

#     def sample(self) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], 
#                              List[tf.Tensor], List[tf.Tensor]]:
#         indices = np.random.randint(0, self.size, size=self.batch_size)
        
#         obs_batch = [
#             tf.convert_to_tensor(self.observations[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         action_batch = [
#             tf.convert_to_tensor(self.actions[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         reward_batch = [
#             tf.convert_to_tensor(self.rewards[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         next_obs_batch = [
#             tf.convert_to_tensor(self.next_observations[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
#         done_batch = [
#             tf.convert_to_tensor(self.dones[i][indices], dtype=tf.float32)
#             for i in range(self.n_agents)
#         ]
        
#         return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

#     def ready(self) -> bool:
#         return self.size >= self.batch_size

#     def clear(self) -> None:
#         self.pointer = 0
#         self.size = 0
#         for i in range(self.n_agents):
#             self.observations[i].fill(0)
#             self.actions[i].fill(0)
#             self.rewards[i].fill(0)
#             self.next_observations[i].fill(0)
#             self.dones[i].fill(0)

# class Actor(tf.keras.Model):
#     def __init__(self, action_dim, hidden_sizes=(256, 256)):
#         super(Actor, self).__init__()
#         self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
#         self.output_layer = Dense(action_dim, activation='softmax')
        
#     def call(self, state):
#         x = state
#         for layer in self.hidden_layers:
#             x = layer(x)
#         action_probs = self.output_layer(x)
#         return action_probs
    
#     def sample_action(self, state):
#         action_probs = self.call(state)
#         action = tf.random.categorical(tf.math.log(action_probs), 1)
#         return action, action_probs

# class DiscreteMultiHeadedCritic(tf.keras.Model):
#     def __init__(self, obs_dim, n_actions, n_agents, hidden_sizes=(256, 256)):
#         super(DiscreteMultiHeadedCritic, self).__init__()
#         self.n_agents = n_agents
#         self.n_actions = n_actions
        
#         # Shared feature extraction layers
#         self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
        
#         # Single output layer with n_agents * n_actions outputs
#         self.output_layer = Dense(n_agents * n_actions)
        
#         # Reshape layer to get proper dimensions
#         self.reshape = Reshape((n_agents, n_actions))
        
#     def call(self, states):
#         # Process states through shared layers
#         x = states
#         for layer in self.hidden_layers:
#             x = layer(x)
        
#         # Get Q-values for all actions of all agents
#         q_values = self.output_layer(x)
        
#         # Reshape to [batch_size, n_agents, n_actions]
#         q_values = self.reshape(q_values)
        
#         return q_values

# class MADDPGAgent:
#     def __init__(self, n_agents, obs_dims, n_actions, gamma=0.99, polyak=0.995, 
#                  lr_actor=3e-4, lr_critic=3e-4, critic_hidden_sizes=(256,256), 
#                  actor_hidden_sizes=(64,64), verbose=True, name="MADDPG"):
#         self.n_agents = n_agents
#         self.n_actions = n_actions
#         self.obs_dims = obs_dims
#         self.gamma = gamma
#         self.polyak = polyak
        
#         # Initialize actors and their targets
#         self.actors = []
#         self.target_actors = []
#         self.pi_optimizers = []
        
#         # Initialize single shared critic and target
#         self.critic = DiscreteMultiHeadedCritic(sum(obs_dims), n_actions, n_agents, 
#                                                critic_hidden_sizes)
#         self.target_critic = DiscreteMultiHeadedCritic(sum(obs_dims), n_actions, n_agents,
#                                                       critic_hidden_sizes)
#         self.q_optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_critic)

#         self.episodes = 0
#         self.name = name
#         if verbose:
#             log_dir = f"../logs/{self.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
#             self.logger = tf.summary.create_file_writer(log_dir)
        
#         for i in range(n_agents):
#             self.actors.append(Actor(n_actions, actor_hidden_sizes))
#             self.target_actors.append(Actor(n_actions, actor_hidden_sizes))
#             self.pi_optimizers.append(tf.keras.optimizers.AdamW(learning_rate=lr_actor))
#             self._initialize_networks(i)
    
#     def _initialize_networks(self, agent_idx):
#         dummy_state = tf.zeros((1, self.obs_dims[agent_idx]))
#         dummy_full_state = tf.zeros((1, sum(self.obs_dims)))
        
#         # Initialize actor networks
#         self.actors[agent_idx](dummy_state)
#         self.target_actors[agent_idx](dummy_state)
        
#         # Initialize critic if this is the first agent
#         if agent_idx == 0:
#             self.critic(dummy_full_state)
#             self.target_critic(dummy_full_state)
#             self.update_target_critic(tau=1.0)
        
#         self.update_target_actor(agent_idx, tau=1.0)
    
#     def update_target_actor(self, agent_idx, tau):
#         for source, target in zip(self.actors[agent_idx].variables, 
#                                 self.target_actors[agent_idx].variables):
#             target.assign(tau * source + (1 - tau) * target)
    
#     def update_target_critic(self, tau):
#         for source, target in zip(self.critic.variables, self.target_critic.variables):
#             target.assign(tau * source + (1 - tau) * target)
    
#     @tf.function
#     def train_step(self, states, actions, rewards, next_states, dones):
#         # Update critic
#         critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
#         # Update actors
#         actor_losses = []
#         for i in range(self.n_agents):
#             actor_loss = self._update_actor(i, states)
#             actor_losses.append(actor_loss)
#             self.update_target_actor(i, self.polyak)
        
#         # Update target critic
#         self.update_target_critic(self.polyak)
        
#         return critic_loss, actor_losses
    
#     def _update_critic(self, states, actions, rewards, next_states, dones):
#         with tf.GradientTape() as tape:
#             # Get target actions probabilities for next states
#             target_action_probs = [target_actor(next_state) 
#                                  for target_actor, next_state 
#                                  in zip(self.target_actors, next_states)]
            
#             # Concatenate all next states
#             next_states_concat = tf.concat(next_states, axis=1)
            
#             # Get Q-values for all actions in next state
#             target_q_values = self.target_critic(next_states_concat)  # [batch, n_agents, n_actions]
            
#             # Calculate expected Q-values using target policy probabilities
#             expected_target_q = []
#             for i in range(self.n_agents):
#                 # Multiply Q-values by action probabilities and sum
#                 agent_target_q = tf.reduce_sum(
#                     target_q_values[:, i, :] * target_action_probs[i], 
#                     axis=1, keepdims=True
#                 )
#                 expected_target_q.append(agent_target_q)
            
#             # Calculate current Q-values for taken actions
#             states_concat = tf.concat(states, axis=1)
#             current_q_all = self.critic(states_concat)  # [batch, n_agents, n_actions]
            
#             # Get Q-values for taken actions
#             current_q_taken = []
#             for i in range(self.n_agents):
#                 # Convert one-hot actions to indices
#                 action_indices = tf.argmax(actions[i], axis=1)
#                 agent_q = tf.gather(current_q_all[:, i, :], action_indices, batch_dims=1)
#                 current_q_taken.append(tf.expand_dims(agent_q, 1))
            
#             # Calculate critic loss for each agent
#             critic_losses = []
#             for i in range(self.n_agents):
#                 target_q = rewards[i] + self.gamma * expected_target_q[i] * (1 - dones[i])
#                 critic_losses.append(tf.reduce_mean(tf.square(target_q - current_q_taken[i])))
            
#             # Total critic loss
#             total_critic_loss = tf.add_n(critic_losses)
        
#         # Update critic
#         critic_grads = tape.gradient(total_critic_loss, self.critic.trainable_variables)
#         self.q_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
#         return total_critic_loss
    
#     def _update_actor(self, agent_idx, states):
#         with tf.GradientTape() as tape:
#             # Get current action probabilities for all agents
#             current_action_probs = []
#             for i, (actor, state) in enumerate(zip(self.actors, states)):
#                 if i == agent_idx:
#                     action_prob = actor(state)
#                 else:
#                     action_prob = tf.stop_gradient(actor(state))
#                 current_action_probs.append(action_prob)
            
#             # Get Q-values for all actions
#             states_concat = tf.concat(states, axis=1)
#             q_values_all = self.critic(states_concat)  # [batch, n_agents, n_actions]
            
#             # Calculate expected Q-value under current policy
#             agent_q_values = q_values_all[:, agent_idx, :]  # [batch, n_actions]
#             expected_q = tf.reduce_sum(agent_q_values * current_action_probs[agent_idx], axis=1)
            
#             # Actor loss is negative of expected Q-value
#             actor_loss = -tf.reduce_mean(expected_q)
        
#         # Update actor
#         actor_grads = tape.gradient(actor_loss, self.actors[agent_idx].trainable_variables)
#         self.pi_optimizers[agent_idx].apply_gradients(
#             zip(actor_grads, self.actors[agent_idx].trainable_variables)
#         )
        
#         return actor_loss
    
#     def log_episode(self, stats):
#         with self.logger.as_default():
#             for key, value in stats.items():
#                 tf.summary.scalar(key, value, step=self.episodes)
#                 self.logger.flush()
#         self.episodes += 1

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from typing import List, Tuple

class MADDPGReplayBuffer:
    def __init__(self, obs_dims: List[int], act_dims: List[int], 
                 size: int, n_agents: int, batch_size: int = 256):
        """
        Initialize a replay buffer for MADDPG.
        
        Args:
            obs_dims: List of observation dimensions for each agent
            act_dims: List of action dimensions for each agent
            size: Maximum size of the buffer
            n_agents: Number of agents
            batch_size: Size of batches to sample
        """
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.size = size
        self.n_agents = n_agents
        self.batch_size = batch_size
        
        # Initialize buffers
        self.observations = [
            np.zeros((size, dim), dtype=np.float32) 
            for dim in obs_dims
        ]
        self.actions = [
            np.zeros((size, dim), dtype=np.float32) 
            for dim in act_dims
        ]
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.next_observations = [
            np.zeros((size, dim), dtype=np.float32) 
            for dim in obs_dims
        ]
        self.dones = np.zeros((size,), dtype=np.float32)
        
        self.ptr = 0
        self.current_size = 0
        
    def store(self, obs: List[np.ndarray], acts: List[np.ndarray], 
              rew: float, next_obs: List[np.ndarray], done: bool) -> None:
        """Store a transition in the buffer"""
        for i in range(self.n_agents):
            self.observations[i][self.ptr] = obs[i]
            self.actions[i][self.ptr] = acts[i]
            self.next_observations[i][self.ptr] = next_obs[i]
        
        self.rewards[self.ptr] = rew
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)
        
    def sample_batch(self):
        """Sample a batch of experiences"""
        idxs = np.random.randint(0, self.current_size, size=self.batch_size)
        
        batch = {
            'observations': [tf.convert_to_tensor(obs[idxs], dtype=tf.float32) 
                           for obs in self.observations],
            'actions': [tf.convert_to_tensor(act[idxs], dtype=tf.float32) 
                       for act in self.actions],
            'rewards': tf.convert_to_tensor(self.rewards[idxs], dtype=tf.float32),
            'next_observations': [tf.convert_to_tensor(next_obs[idxs], dtype=tf.float32) 
                                for next_obs in self.next_observations],
            'dones': tf.convert_to_tensor(self.dones[idxs], dtype=tf.float32)
        }
        
        return batch
    
    def ready(self) -> bool:
        """Check if enough samples are available"""
        return self.current_size >= self.batch_size
    
class OUNoise:
    """Ornstein-Uhlenbeck process noise generator"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()
    
    def reset(self):
        """Reset the internal state (noise) to mean"""
        self.state = np.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state

class ContinuousActor(tf.keras.Model):
    def __init__(self, action_dim, action_high, hidden_sizes=(256, 256)):
        """
        Initialize continuous actor network
        
        Args:
            action_dim: Dimension of continuous action space
            action_high: Upper bound of action space
            hidden_sizes: Sizes of hidden layers
        """
        super(ContinuousActor, self).__init__()
        self.action_high = action_high
        self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
        self.output_layer = Dense(action_dim, activation='tanh')
        
    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        # Scale tanh output to action space
        action = self.output_layer(x) * self.action_high
        return action
    
    def sample_action(self, state, noise=None):
        """Sample action with optional exploration noise"""
        action = self.call(state)
        if noise is not None:
            action = tf.clip_by_value(action + noise, -self.action_high, self.action_high)
        return action

class ContinuousCritic(tf.keras.Model):
    def __init__(self, obs_dim, action_dim, n_agents, hidden_sizes=(256, 256)):
        super(ContinuousCritic, self).__init__()
        self.hidden_layers = [Dense(size, activation='relu') for size in hidden_sizes]
        self.output_layer = Dense(1)  # Single Q-value output
        
    def call(self, inputs):
        states, actions = inputs
        # Concatenate states and actions
        x = tf.concat([states, actions], axis=-1)
        
        for layer in self.hidden_layers:
            x = layer(x)
        q_value = self.output_layer(x)
        return q_value

class ContinuousMADDPGAgent:
    def __init__(self, n_agents, obs_dims, action_dims, action_high,
                 gamma=0.95, polyak=0.99, lr_actor=1e-3, lr_critic=1e-3,
                 critic_hidden_sizes=(256, 128), actor_hidden_sizes=(64, 64),
                 verbose=True, name="ContinuousMADDPG", update_freq=1):
        """
        Initialize Continuous MADDPG agent
        
        Args:
            n_agents: Number of agents
            obs_dims: List of observation dimensions for each agent
            action_dims: List of action dimensions for each agent
            action_high: Upper bound of action space
            gamma: Discount factor
            polyak: Soft update coefficient
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            critic_hidden_sizes: Hidden layer sizes for critic
            actor_hidden_sizes: Hidden layer sizes for actor
            verbose: Whether to log training statistics
            name: Name of the agent
        """
        self.n_agents = n_agents
        self.action_dims = action_dims
        self.obs_dims = obs_dims
        self.gamma = gamma
        self.polyak = polyak
        self.update_freq = update_freq
        self.step_counter = 0
        
        # Initialize networks and optimizers for each agent
        self.actors = []
        self.target_actors = []
        self.critics = []
        self.target_critics = []
        self.pi_optimizers = []
        self.q_optimizers = []
        
        # Initialize noise processes for exploration
        self.noise_processes = [
            OUNoise(dim) for dim in action_dims
        ]

        self.episodes = 0
        self.name = name
        if verbose:
            log_dir = f"../logs/{self.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.logger = tf.summary.create_file_writer(log_dir)
        
        for i in range(n_agents):
            # Create actor networks
            self.actors.append(ContinuousActor(action_dims[i], action_high, actor_hidden_sizes))
            self.target_actors.append(ContinuousActor(action_dims[i], action_high, actor_hidden_sizes))
            
            # Create critic networks
            total_action_dim = sum(action_dims)
            self.critics.append(ContinuousCritic(sum(obs_dims), total_action_dim, n_agents, 
                                               critic_hidden_sizes))
            self.target_critics.append(ContinuousCritic(sum(obs_dims), total_action_dim, n_agents,
                                                      critic_hidden_sizes))
            
            # Create optimizers
            self.pi_optimizers.append(tf.keras.optimizers.AdamW(learning_rate=lr_actor))
            self.q_optimizers.append(tf.keras.optimizers.AdamW(learning_rate=lr_critic))
            
            # Initialize networks with dummy data
            self._initialize_networks(i)
    
    def _initialize_networks(self, agent_idx):
        """Initialize networks with dummy data"""
        dummy_state = tf.zeros((1, self.obs_dims[agent_idx]))
        dummy_full_state = tf.zeros((1, sum(self.obs_dims)))
        dummy_actions = tf.zeros((1, sum(self.action_dims)))
        
        # Initialize actor networks
        self.actors[agent_idx](dummy_state)
        self.target_actors[agent_idx](dummy_state)
        
        # Initialize critic networks
        self.critics[agent_idx]([dummy_full_state, dummy_actions])
        self.target_critics[agent_idx]([dummy_full_state, dummy_actions])
        
        # Copy initial weights to targets
        self.update_target_networks(agent_idx, tau=1.0)
    
    def update_target_networks(self, agent_idx, tau):
        """Update target networks using polyak averaging"""
        # Update actor target
        for source, target in zip(self.actors[agent_idx].variables, 
                                self.target_actors[agent_idx].variables):
            target.assign(tau * source + (1 - tau) * target)
        
        # Update critic target
        for source, target in zip(self.critics[agent_idx].variables, 
                                self.target_critics[agent_idx].variables):
            target.assign(tau * source + (1 - tau) * target)
    
    def get_action(self, obs, add_noise=True):
        """Get actions for all agents"""
        actions = []
        for i, (actor, noise_process) in enumerate(zip(self.actors, self.noise_processes)):
            state = tf.expand_dims(tf.convert_to_tensor(obs[i], dtype=tf.float32), 0)
            noise = noise_process.sample() if add_noise else None
            action = actor.sample_action(state, noise)
            # actions.append(action[0].numpy())
            discrete_action = tf.argmax(action, axis=-1).numpy()

            # Example 2: Using Thresholding (for continuous outputs mapped to discrete bins)
            # discrete_action = tf.cast(tf.round((continuous_action + 1) / 2), tf.int32).numpy()
            # This maps continuous values in [-1, 1] to discrete bins [0, 1]

            # Append the first (and only) action since we use batch dimension
            actions.append(discrete_action[0])
        return actions
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Training step for all agents"""
        critic_losses = []
        actor_losses = []
        
        if self.step_counter % self.update_freq ==0:
            for i in range(self.n_agents):
                critic_loss = self._update_critic(i, states, actions, rewards[i], next_states, dones[i])
                actor_loss = self._update_actor(i, states)
                self.update_target_networks(i, self.polyak)
                
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
            
            return critic_losses, actor_losses
    
    def _update_critic(self, agent_idx, states, actions, rewards, next_states, dones):
        """Update critic for a specific agent"""
        with tf.GradientTape() as tape:
            # Get target actions for next states
            target_actions = [target_actor(next_state) 
                            for target_actor, next_state 
                            in zip(self.target_actors, next_states)]
            
            # Calculate target Q-value
            target_q = self.target_critics[agent_idx](
                [tf.concat(next_states, axis=1), 
                 tf.concat(target_actions, axis=1)]
            )
            
            target_q = rewards + self.gamma * target_q * (1 - dones)
            
            # Calculate current Q-value
            current_q = self.critics[agent_idx](
                [tf.concat(states, axis=1), 
                 tf.concat(actions, axis=1)]
            )
            
            # Calculate critic loss
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        # Update critic
        critic_grads = tape.gradient(critic_loss, self.critics[agent_idx].trainable_variables)
        self.q_optimizers[agent_idx].apply_gradients(
            zip(critic_grads, self.critics[agent_idx].trainable_variables)
        )
        
        return critic_loss
    
    def _update_actor(self, agent_idx, states):
        """Update actor for a specific agent"""
        with tf.GradientTape() as tape:
            # Get current actions for all agents
            current_actions = []
            for i, (actor, state) in enumerate(zip(self.actors, states)):
                if i == agent_idx:
                    action = actor(state)
                else:
                    action = tf.stop_gradient(actor(state))
                current_actions.append(action)
            
            # Calculate actor loss (negative Q value)
            actor_loss = -tf.reduce_mean(
                self.critics[agent_idx](
                    [tf.concat(states, axis=1), 
                     tf.concat(current_actions, axis=1)]
                )
            )
        
        # Update actor
        actor_grads = tape.gradient(actor_loss, self.actors[agent_idx].trainable_variables)
        self.pi_optimizers[agent_idx].apply_gradients(
            zip(actor_grads, self.actors[agent_idx].trainable_variables)
        )
        
        return actor_loss
    
    def reset_noise(self):
        """Reset noise processes for all agents"""
        for noise_process in self.noise_processes:
            noise_process.reset()
    
    def log_episode(self, stats):
        """Log training statistics"""
        with self.logger.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.episodes)
                self.logger.flush()
        self.episodes += 1

# import datetime
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, LayerNormalization, BatchNormalization
# from typing import List, Tuple

# class MADDPGReplayBuffer:
#     def __init__(self, obs_dims: List[int], act_dims: List[int], 
#                  size: int, n_agents: int, batch_size: int = 256):
#         self.obs_dims = obs_dims
#         self.act_dims = act_dims
#         self.size = size
#         self.n_agents = n_agents
#         self.batch_size = batch_size
        
#         # Use pre-allocated numpy arrays with proper data types
#         self.observations = [
#             np.zeros((size, dim), dtype=np.float32) 
#             for dim in obs_dims
#         ]
#         self.actions = [
#             np.zeros((size, dim), dtype=np.float32) 
#             for dim in act_dims
#         ]
#         self.rewards = np.zeros((size, n_agents), dtype=np.float32)  # Changed to store per-agent rewards
#         self.next_observations = [
#             np.zeros((size, dim), dtype=np.float32) 
#             for dim in obs_dims
#         ]
#         self.dones = np.zeros((size,), dtype=np.float32)
        
#         self.ptr = 0
#         self.current_size = 0
        
#         # Pre-allocate indices array for faster sampling
#         self.indices = np.arange(size)
        
#     # @tf.function(experimental_relax_shapes=True)
#     def store(self, obs: List[np.ndarray], acts: List[np.ndarray], 
#               rews: List[float], next_obs: List[np.ndarray], done: bool) -> None:
#         """Store a transition in the buffer with vectorized operations"""
#         for i in range(self.n_agents):
#             self.observations[i][self.ptr] = obs[i]
#             self.actions[i][self.ptr] = acts[i]
#             self.next_observations[i][self.ptr] = next_obs[i]
#             self.rewards[self.ptr, i] = rews[i]  # Store per-agent rewards
        
#         self.dones[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.size
#         self.current_size = min(self.current_size + 1, self.size)
        
#     def sample_batch(self):
#         """Sample a batch of experiences using vectorized operations"""
#         idxs = np.random.choice(self.indices[:self.current_size], 
#                               size=self.batch_size, replace=False)
        
#         batch = {
#             'observations': [tf.constant(obs[idxs], dtype=tf.float32) 
#                            for obs in self.observations],
#             'actions': [tf.constant(act[idxs], dtype=tf.float32) 
#                        for act in self.actions],
#             'rewards': tf.constant(self.rewards[idxs], dtype=tf.float32),
#             'next_observations': [tf.constant(next_obs[idxs], dtype=tf.float32) 
#                                 for next_obs in self.next_observations],
#             'dones': tf.constant(self.dones[idxs], dtype=tf.float32)
#         }
        
#         return batch

# class ContinuousActor(tf.keras.Model):
#     def __init__(self, action_dim, action_high, hidden_sizes=(400, 300)):
#         super(ContinuousActor, self).__init__()
#         self.action_high = action_high
        
#         # Use larger networks with normalization layers
#         self.hidden_layers = []
#         for size in hidden_sizes:
#             self.hidden_layers.extend([
#                 Dense(size, kernel_initializer='glorot_uniform'),
#                 LayerNormalization(),
#                 tf.keras.layers.Activation('relu')
#             ])
            
#         self.output_layer = Dense(action_dim, activation='tanh',
#                                 kernel_initializer='glorot_uniform')
        
#     @tf.function(experimental_relax_shapes=True)
#     def call(self, state):
#         x = state
#         for layer in self.hidden_layers:
#             x = layer(x)
#         return self.output_layer(x) * self.action_high

# class ContinuousCritic(tf.keras.Model):
#     def __init__(self, obs_dim, action_dim, n_agents, hidden_sizes=(400, 300)):
#         super(ContinuousCritic, self).__init__()
        
#         self.hidden_layers = []
#         for size in hidden_sizes:
#             self.hidden_layers.extend([
#                 Dense(size, kernel_initializer='glorot_uniform'),
#                 LayerNormalization(),
#                 tf.keras.layers.Activation('relu')
#             ])
            
#         self.output_layer = Dense(1, kernel_initializer='glorot_uniform')
    
#     @tf.function(experimental_relax_shapes=True)
#     def call(self, inputs):
#         states, actions = inputs
#         x = tf.concat([states, actions], axis=-1)
        
#         for layer in self.hidden_layers:
#             x = layer(x)
#         return self.output_layer(x)

# class ContinuousMADDPGAgent:
#     def __init__(self, n_agents, obs_dims, action_dims, action_high,
#                  gamma=0.99, polyak=0.995, lr_actor=3e-4, lr_critic=3e-4,
#                  critic_hidden_sizes=(400, 300), actor_hidden_sizes=(400, 300),
#                  update_freq=2, verbose=True, name="ContinuousMADDPG"):
        
#         self.n_agents = n_agents
#         self.action_dims = action_dims
#         self.action_high = action_high
#         self.obs_dims = obs_dims
#         self.gamma = gamma
#         self.polyak = polyak
#         self.update_freq = update_freq
#         self.step_counter = 0
        
#         # Enable mixed precision training
#         policy = tf.keras.mixed_precision.Policy('mixed_float16')
#         tf.keras.mixed_precision.set_global_policy(policy)
        
#         # Initialize networks with optimized architectures
#         self.actors = []
#         self.target_actors = []
#         self.critics = []
#         self.target_critics = []
        
#         # Use AMSGrad optimizer with gradient clipping
#         optimizer_config = {
#             'learning_rate': lr_actor,
#             'amsgrad': True,
#             'clipnorm': 1.0,
#             'weight_decay': 1e-4
#         }
        
#         self.pi_optimizers = []
#         self.q_optimizers = []
        
#         # Initialize noise processes with decaying noise
#         self.noise_scale = 1.0
#         self.noise_decay = 0.9995
#         self.noise_processes = [
#             OUNoise(size=dim, sigma=0.3) for dim in action_dims
#         ]
        
#         for i in range(n_agents):
#             # Create networks with optimized architectures
#             self.actors.append(ContinuousActor(action_dims[i], action_high, actor_hidden_sizes))
#             self.target_actors.append(ContinuousActor(action_dims[i], action_high, actor_hidden_sizes))
            
#             total_action_dim = sum(action_dims)
#             self.critics.append(ContinuousCritic(sum(obs_dims), total_action_dim, 
#                                                n_agents, critic_hidden_sizes))
#             self.target_critics.append(ContinuousCritic(sum(obs_dims), total_action_dim,
#                                                       n_agents, critic_hidden_sizes))
            
#             # Initialize optimizers with AMSGrad and gradient clipping
#             self.pi_optimizers.append(tf.keras.optimizers.AdamW(**optimizer_config))
#             self.q_optimizers.append(tf.keras.optimizers.AdamW(**optimizer_config))
            
#             self._initialize_networks(i)
        
#         # Initialize logger
#         if verbose:
#             log_dir = f"logs/{name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
#             self.logger = tf.summary.create_file_writer(log_dir)

#     def _initialize_networks(self, agent_idx):
#         """Initialize networks with dummy data"""
#         dummy_state = tf.zeros((1, self.obs_dims[agent_idx]))
#         dummy_full_state = tf.zeros((1, sum(self.obs_dims)))
#         dummy_actions = tf.zeros((1, sum(self.action_dims)))
        
#         # Initialize actor networks
#         self.actors[agent_idx](dummy_state)
#         self.target_actors[agent_idx](dummy_state)
        
#         # Initialize critic networks
#         self.critics[agent_idx]([dummy_full_state, dummy_actions])
#         self.target_critics[agent_idx]([dummy_full_state, dummy_actions])
        
#         # Copy initial weights to targets
#         self.update_target_networks(agent_idx, tau=1.0)
    
#     @tf.function(experimental_relax_shapes=True)
#     def update_target_networks(self, agent_idx, tau):
#         """Update target networks using vectorized operations"""
#         for source, target in zip(self.actors[agent_idx].variables, 
#                                 self.target_actors[agent_idx].variables):
#             target.assign(tau * source + (1 - tau) * target)
        
#         for source, target in zip(self.critics[agent_idx].variables, 
#                                 self.target_critics[agent_idx].variables):
#             target.assign(tau * source + (1 - tau) * target)
    
#     @tf.function(experimental_relax_shapes=True)
#     def get_action(self, obs, add_noise=True):
#         """Get actions for all agents with optional noise"""
#         actions = []
#         for i, (actor, noise_process) in enumerate(zip(self.actors, self.noise_processes)):
#             state = tf.expand_dims(tf.convert_to_tensor(obs[i], dtype=tf.float32), 0)
#             action = actor(state)
            
#             if add_noise:
#                 noise = noise_process.sample() * self.noise_scale
#                 action = tf.clip_by_value(action + noise, -self.action_high, self.action_high)
            
#             actions.append(action[0])
        
#         # Decay exploration noise
#         self.noise_scale *= self.noise_decay
#         return actions
    
#     def reset_noise(self):
#         """Reset noise processes for all agents"""
#         for noise_process in self.noise_processes:
#             noise_process.reset()
    
#     @tf.function(experimental_relax_shapes=True)
#     def train_step(self, states, actions, rewards, next_states, dones):
#         """Optimized training step with less frequent target updates"""
#         self.step_counter += 1
        
#         critic_losses = []
#         actor_losses = []
        
#         # Update critics first
#         for i in range(self.n_agents):
#             critic_loss = self._update_critic(i, states, actions, rewards[:, i], next_states, dones)
#             critic_losses.append(critic_loss)
        
#         # Update actors and targets less frequently
#         if self.step_counter % self.update_freq == 0:
#             for i in range(self.n_agents):
#                 actor_loss = self._update_actor(i, states)
#                 self.update_target_networks(i, self.polyak)
#                 actor_losses.append(actor_loss)
        
#         return critic_losses, actor_losses