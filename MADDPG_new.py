import tensorflow as tf
import numpy as np
import random
import datetime
from tensorflow.keras import layers, Model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def store_transition(self, states, actions, rewards, next_states, dones):
        # Store each transition in the replay buffer
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)

        self.buffer[self.position] = (states, actions, rewards, next_states, dones)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)

        # Extract individual elements from the batch
        states_batch = [b[0] for b in batch]
        actions_batch = [b[1] for b in batch]
        rewards_batch = np.array([b[2] for b in batch], dtype=np.float32)
        next_states_batch = [b[3] for b in batch]
        dones_batch = np.array([b[4] for b in batch], dtype=np.float32)

        # Handle non-homogeneous states
        states_batch = [np.array([s[i] for s in states_batch]) for i in range(len(states_batch[0]))]
        next_states_batch = [np.array([ns[i] for ns in next_states_batch]) for i in range(len(next_states_batch[0]))]

        # Handling actions: if actions are not homogeneous, treat them similarly, though this example assumes they are
        actions_batch = [np.array([a[i] for a in actions_batch]) for i in range(len(actions_batch[0]))]

        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch

    def __len__(self):
        return len(self.buffer)
    
class Actor(Model):
    def __init__(self, num_actions, hidden_units=[64, 64], name='Actor'):
        super(Actor, self).__init__(name=name)
        self.fc1 = layers.Dense(hidden_units[0], activation='relu')
        self.fc2 = layers.Dense(hidden_units[1], activation='relu')
        self.output_layer = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        logits = self.output_layer(x)
        # Gumbel-softmax for action sampling
        u = tf.random.uniform(tf.shape(logits))
        actions = tf.nn.softmax(logits - tf.math.log(-tf.math.log(u)), axis=-1)

        return actions

class Critic(Model):
    def __init__(self, hidden_units=[256, 128], name='Critic'):
        super(Critic, self).__init__(name=name)
        self.fc1 = layers.Dense(hidden_units[0], activation='relu')
        self.fc2 = layers.Dense(hidden_units[1], activation='relu')
        self.output_layer = layers.Dense(1, activation='linear')

    def call(self, state_inputs, action_inputs):
        # Assuming state_inputs and action_inputs are combined appropriately before passing to the Critic
        x = tf.concat([state_inputs, action_inputs], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output_layer(x)
    
class MADDPGAgent:
    def __init__(self, obs_dim, num_actions, num_agents, lr=0.01, tau=0.01, gamma=0.95, name="MADDPG", verbose=True):
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.tau = tau    # Target network update rate
        self.gamma = gamma # Discount factor
        self.lr = lr
        self.obs_dim = obs_dim

        # Initialize actor and critic networks for each agent
        self.actors = [Actor(num_actions) for _ in range(num_agents)]
        self.critics = [Critic() for _ in range(num_agents)]

        # Initialize target networks
        self.target_actors = [Actor(num_actions) for _ in range(num_agents)]
        self.target_critics = [Critic() for _ in range(num_agents)]

        # Initialize optimizers
        self.actor_optimizers = [tf.keras.optimizers.Adam(learning_rate=lr) for _ in range(num_agents)]
        self.critic_optimizers = [tf.keras.optimizers.Adam(learning_rate=lr) for _ in range(num_agents)]

        self.episodes = 0
        self.name = name
        if verbose:
            log_dir = f"../logs/{self.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.logger = tf.summary.create_file_writer(log_dir)

        # Copy initial weights to target networks
        for i in range(num_agents):
            dummy_state = tf.zeros((1, obs_dim[i]))
            dummy_full_state = tf.zeros((1, sum(obs_dim)))
            dummy_actions = tf.zeros((1, num_agents*num_actions))
            
            # Initialize actor networks
            self.actors[i](dummy_state)
            self.target_actors[i](dummy_state)
            
            # Initialize critic networks
            self.critics[i](dummy_full_state, dummy_actions)
            self.target_critics[i](dummy_full_state, dummy_actions)
            self.update_network_parameters(self.actors[i], self.target_actors[i], 1.0)
            self.update_network_parameters(self.critics[i], self.target_critics[i], 1.0)

    def reset_weights(self):
        # Initialize actor and critic networks for each agent
        self.actors = [Actor(self.num_actions) for _ in range(self.num_agents)]
        self.critics = [Critic() for _ in range(self.num_agents)]

        # Initialize target networks
        self.target_actors = [Actor(self.num_actions) for _ in range(self.num_agents)]
        self.target_critics = [Critic() for _ in range(self.num_agents)]

        # Initialize optimizers
        self.actor_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.lr) for _ in range(self.num_agents)]
        self.critic_optimizers = [tf.keras.optimizers.Adam(learning_rate=self.lr) for _ in range(self.num_agents)]

        self.episodes = 0

        for i in range(self.num_agents):
            dummy_state = tf.zeros((1, self.obs_dim[i]))
            dummy_full_state = tf.zeros((1, sum(self.obs_dim)))
            dummy_actions = tf.zeros((1, self.num_agents*self.num_actions))
            
            # Initialize actor networks
            self.actors[i](dummy_state)
            self.target_actors[i](dummy_state)
            
            # Initialize critic networks
            self.critics[i](dummy_full_state, dummy_actions)
            self.target_critics[i](dummy_full_state, dummy_actions)
            self.update_network_parameters(self.actors[i], self.target_actors[i], 1.0)
            self.update_network_parameters(self.critics[i], self.target_critics[i], 1.0)
        
        # Clear TensorFlow's memory
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        import gc
        gc.collect()

    def select_action(self, state, agent_index, explore=True, epsilon=0.1):
        # Convert state to tensor for the model
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Get action probabilities from the actor network
        action_probs = self.actors[agent_index](state)
        
        # Select the action with the highest probability (deterministic action)
        action = tf.argmax(action_probs[0]).numpy()

        if explore:
            if np.random.rand() < epsilon:
                # With probability epsilon, choose a random action for exploration
                action = np.random.randint(self.num_actions)
                action_probs = tf.one_hot([action],5)
                
        return action_probs[0], int(action)

    def log_episode(self, stats):
        """Log statistics."""
        with self.logger.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.episodes)
                self.logger.flush()
        self.episodes += 1
        
    def update_network_parameters(self, source_model, target_model, tau):
        for (source_params, target_params) in zip(source_model.trainable_variables, target_model.trainable_variables):
            target_params.assign(tau * source_params + (1.0 - tau) * target_params)

    def update_target_networks(self):
        # Update target networks using tau
        for i in range(self.num_agents):
            self.update_network_parameters(self.actors[i], self.target_actors[i], self.tau)
            self.update_network_parameters(self.critics[i], self.target_critics[i], self.tau)

    def train(self, replay_buffer):
        if len(replay_buffer) < replay_buffer.batch_size:
            return
        else:
            self.learn(*replay_buffer.sample())

    @tf.function
    def learn(self, states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch):
        for agent_idx in range(self.num_agents):
            with tf.GradientTape() as tape:
                target_actions = [
                    self.target_actors[i](next_states_batch[i]) for i in range(self.num_agents)
                ]
                target_actions = tf.concat(target_actions, axis=-1)

                stacked_next_states = tf.concat(next_states_batch, axis=-1)
                q_next = self.target_critics[agent_idx](stacked_next_states, target_actions)
                q_next = rewards_batch[:, agent_idx] + self.gamma * (1 - dones_batch[:, agent_idx]) * tf.squeeze(q_next)

                stacked_states = tf.concat(states_batch, axis=-1)
                q_current = self.critics[agent_idx](stacked_states, tf.concat(actions_batch, axis=-1))
                critic_loss = tf.reduce_mean(tf.square(q_next - tf.squeeze(q_current)))

            critic_grad = tape.gradient(critic_loss, self.critics[agent_idx].trainable_variables)
            self.critic_optimizers[agent_idx].apply_gradients(zip(critic_grad, self.critics[agent_idx].trainable_variables))

            with tf.GradientTape() as tape:
                current_actions = [
                    self.actors[i](states_batch[i]) if i == agent_idx else actions_batch[i]
                    for i in range(self.num_agents)
                ]
                current_actions = tf.concat(current_actions, axis=-1)

                q_eval = self.critics[agent_idx](stacked_states, current_actions)
                actor_loss = -tf.reduce_mean(q_eval)

            actor_grad = tape.gradient(actor_loss, self.actors[agent_idx].trainable_variables)
            self.actor_optimizers[agent_idx].apply_gradients(zip(actor_grad, self.actors[agent_idx].trainable_variables))

        # Update the target networks
        self.update_target_networks()