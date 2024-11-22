import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class PPOMemory:
    def __init__(self, batch_size):
        """
        Initializes the memory for PPO with a given batch size.
        
        :param minibatch_size: The size of each batch used in training.
        """
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def get_transitions(self):
        """
        Generates shuffled batches of experiences for training.

        :return: A tuple containing arrays of states, actions, probabilities, 
                 values, rewards, dones, and the list of batch indices.
        """
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        probs = np.array(self.probs, dtype=np.float32)
        vals = np.array(self.vals, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        return states, actions, probs, vals, rewards, dones

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        Stores a single transition in memory.

        :param state: The observed state.
        :param action: The action taken.
        :param probs: The probabilities associated with the taken action.
        :param vals: The value estimated by the critic.
        :param reward: The reward received after taking the action.
        :param done: Boolean indicating if the episode is finished.
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """
        Clears the memory buffers.
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
    
    def ready(self):
        return len(self.states) >= self.batch_size
    
class ActorNetwork(Model):
    def __init__(self, n_actions, action_dims, hidden_dims=[256,256]):
        super().__init__()
        self.hidden_layers = [Dense(dims, activation='relu') for dims in hidden_dims]
        self.fc3 = Dense(
            n_actions * action_dims
        )
        self.n_actions = n_actions
        self.action_dims = action_dims

    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.fc3(x)
        action_probs = tf.reshape(out, (-1, self.action_dims, self.n_actions))
        action_probs = tf.nn.softmax(action_probs)
        return action_probs

class CriticNetwork(Model):
    def __init__(self, hidden_dims=[256,256]):
        super().__init__()
        self.hidden_layers = [Dense(dims, activation='relu') for dims in hidden_dims]
        self.q = Dense(
            1, activation=None
        )

    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        v = self.q(x)
        return v

class Agent:
    def __init__(self, n_actions, action_dims, name='PPO', gamma=0.99, lr=3e-4,
                 gae_lambda=0.95, policy_clip=0.2, entropy_coef=0.001, 
                 critic_hidden_sizes=[256,256], actor_hidden_sizes=[256,256],
                 batch_size=500, n_epochs=100, chkpt_dir='models/', verbose=True):
        self.name = name
        self.n_actions = n_actions
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.verbose = verbose
        
        self.actor = ActorNetwork(n_actions, action_dims, actor_hidden_sizes)
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        self.critic = CriticNetwork(critic_hidden_sizes)
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        self.memory = PPOMemory(batch_size)

        if verbose:
            log_dir = f"../logs/dissertation-result-ppo-SN-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.logger = tf.summary.create_file_writer(log_dir)
        self.episodes = 0

    def store_transition(self, state, action, probs, vals, reward, done):
        """Store a transition in memory."""
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        """Choose an action based on the current observation."""
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        action_probs = self.actor(state)
        value = self.critic(state).numpy()[0]

        dist = tfp.distributions.Categorical(action_probs)
        sampled_action = tf.reshape(dist.sample(), (-1,))
        log_probs = tf.reshape(dist.log_prob(sampled_action), (-1,))
        return sampled_action.numpy(), log_probs, value
    
    @tf.function
    def learn(self, state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr):
        """Train the actor and critic networks."""
        actor_losses = 0
        states = tf.convert_to_tensor(state_arr, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_prob_arr, dtype=tf.float32)
        old_probs = tf.reduce_sum(old_probs, axis=1)
        actions = tf.convert_to_tensor(action_arr, dtype=tf.int32)
        rewards = tf.convert_to_tensor(reward_arr, dtype=tf.float32)
        values = tf.convert_to_tensor(vals_arr, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones_arr, dtype=tf.float32)

        # Calculate advantage using tensor operations
        advantage = tf.zeros_like(rewards)

        # Iterate through time steps in reverse (for better tensor performance)
        gae = 0
        for t in reversed(range(len(reward_arr) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            gae = tf.squeeze(gae)
            # Corrected tensor update to match scalar shape
            advantage = tf.tensor_scatter_nd_update(advantage, [[t]], [gae])

        for _ in range(self.n_epochs):
            with tf.GradientTape(persistent=True) as tape:
                action_probs = self.actor(states)
                dist = tfp.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                critic_value = tf.squeeze(self.critic(states), 1)
                
                new_log_probs = tf.reduce_sum(new_log_probs, axis=1)
                prob_ratio = tf.exp(new_log_probs - old_probs)
                weighted_probs = advantage * prob_ratio
                clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                weighted_clipped_probs = clipped_probs * advantage
                actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_clipped_probs))

                entropy = tf.reduce_mean(dist.entropy())
                actor_loss -= self.entropy_coef * entropy

                returns = advantage + values
                critic_loss = tf.keras.losses.MSE(critic_value, returns)

            actor_losses += actor_loss
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            del tape

    def train(self):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = self.memory.get_transitions()
        self.learn(state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr)
        self.memory.clear_memory()

    def save_models(self):
        """Save actor and critic models."""
        print('... saving models ...')
        self.actor.save(f'model/{self.name}_actor', save_format="tf")
        self.critic.save(f'model/{self.name}_critic', save_format="tf")

    def load_models(self):
        """Load actor and critic models."""
        try:
            print('... loading models ...')
            self.actor = tf.keras.models.load_model(f'model/{self.name}_actor')
            self.critic = tf.keras.models.load_model(f'model/{self.name}_critic')
            print('... Models loaded ...')
        except Exception as e:
            print(e)
            print("Model files failed to load. Using randomly initialized weights.")

    def log_episode(self, stats):
        """Log statistics."""
        with self.logger.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.episodes)
                self.logger.flush()
        self.episodes += 1

# import numpy as np
# import tensorflow as tf
# import tensorflow_probability as tfp
# import datetime
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense

# class PPOMemory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []

#         self.batch_size = batch_size

#     def get_transitions(self):
#         states = np.array(self.states, dtype=np.float32)
#         actions = np.array(self.actions, dtype=np.int32)
#         probs = np.array(self.probs, dtype=np.float32)
#         vals = np.array(self.vals, dtype=np.float32)
#         rewards = np.array(self.rewards, dtype=np.float32)
#         dones = np.array(self.dones, dtype=np.float32)

#         return states, actions, probs, vals, rewards, dones

#     def store_memory(self, state, action, probs, vals, reward, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.probs.append(probs)
#         self.vals.append(vals)
#         self.rewards.append(reward)
#         self.dones.append(done)

#     def clear_memory(self):
#         self.states = []
#         self.probs = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.vals = []
    
#     def ready(self):
#         return len(self.states) >= self.batch_size

# class ActorCriticNetwork(Model):
#     def __init__(self, n_actions, action_dims, fcn_dims=256):
#         super().__init__()
#         self.shared_fc1 = Dense(fcn_dims, activation='relu')
#         self.shared_fc2 = Dense(128, activation='relu')

#         # Actor head (for action probabilities)
#         self.actor_output = Dense(n_actions * action_dims)

#         # Critic head (for value estimation)
#         self.critic_output = Dense(1, activation=None)

#         self.n_actions = n_actions
#         self.action_dims = action_dims

#     def call(self, state):
#         # Shared layers
#         shared_inp = self.shared_fc1(state)
#         shared_inp = self.shared_fc2(shared_inp)

#         # Actor output (softmax probabilities for each action dimension)
#         action_probs = self.actor_output(shared_inp)
#         action_probs = tf.reshape(action_probs, (-1, self.action_dims, self.n_actions))
#         action_probs = tf.nn.softmax(action_probs)

#         # Critic output (state value)
#         state_value = self.critic_output(shared_inp)

#         return action_probs, state_value
    
#     def policy(self, state):
#         # Shared layers
#         shared_inp = self.shared_fc1(state)
#         shared_inp = self.shared_fc2(shared_inp)

#         # Actor output (softmax probabilities for each action dimension)
#         action_probs = self.actor_output(shared_inp)
#         action_probs = tf.reshape(action_probs, (-1, self.action_dims, self.n_actions))
#         action_probs = tf.nn.softmax(action_probs)

#         return action_probs
    
#     def value(self, state):
#         # Shared layers
#         shared_inp = self.shared_fc1(state)
#         shared_inp = self.shared_fc2(shared_inp)

#         # Critic output (state value)
#         state_value = self.critic_output(shared_inp)

#         return state_value

# class Agent:
#     def __init__(self, n_actions, action_dims, name='PPOUnified', gamma=0.99, lr=3e-4,
#                  gae_lambda=0.95, policy_clip=0.2, entropy_coef=0.001, batch_size=500,
#                  n_epochs=100, chkpt_dir='models/', verbose=True):
#         self.name = name
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.policy_clip = policy_clip
#         self.n_epochs = n_epochs
#         self.gae_lambda = gae_lambda
#         self.chkpt_dir = chkpt_dir
#         self.batch_size = batch_size
#         self.entropy_coef = entropy_coef
#         self.verbose = verbose
#         self.episodes = 0
        
#         self.actor_critic = ActorCriticNetwork(n_actions, action_dims)
#         self.actor_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

#         self.memory = PPOMemory(batch_size)

#         if verbose:
#             log_dir = f"logs/{self.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
#             self.logger = tf.summary.create_file_writer(log_dir)
#         self.episodes = 0

#     def store_transition(self, state, action, probs, vals, reward, done):
#         """Store a transition in memory."""
#         self.memory.store_memory(state, action, probs, vals, reward, done)

#     def choose_action(self, observation):
#         """Choose an action based on the current observation."""
#         state = tf.convert_to_tensor([observation], dtype=tf.float32)
#         action_probs, value = self.actor_critic(state)

#         dist = tfp.distributions.Categorical(action_probs)
#         sampled_action = tf.reshape(dist.sample(), (-1,))
#         log_probs = tf.reshape(dist.log_prob(sampled_action), (-1,))
#         return sampled_action.numpy(), log_probs, value.numpy()[0]
    
#     @tf.function
#     def learn(self, state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr):
#         """Train the actor-critic network."""
#         actor_losses = 0
#         states = tf.convert_to_tensor(state_arr, dtype=tf.float32)
#         old_probs = tf.convert_to_tensor(old_prob_arr, dtype=tf.float32)
#         old_probs = tf.reduce_sum(old_probs, axis=1)  # This sums over action dimensions
#         actions = tf.convert_to_tensor(action_arr, dtype=tf.int32)
#         rewards = tf.convert_to_tensor(reward_arr, dtype=tf.float32)
#         values = tf.convert_to_tensor(vals_arr, dtype=tf.float32)
#         dones = tf.convert_to_tensor(dones_arr, dtype=tf.float32)

#         # # Compute discounted rewards (returns)
#         # returns = []
#         # discounted_sum = 0
#         # for reward, done in zip(reversed(rewards.numpy()), reversed(dones.numpy())):
#         #     discounted_sum = reward + (self.gamma * discounted_sum * (1 - done))
#         #     returns.insert(0, discounted_sum)
#         # returns = tf.convert_to_tensor(returns, dtype=tf.float32)

#         # # Calculate advantages
#         # advantages = returns - values.numpy()

#         # Calculate advantage using tensor operations
#         advantages = tf.zeros_like(rewards)

#         # Iterate through time steps in reverse (for better tensor performance)
#         gae = 0
#         for t in reversed(range(len(reward_arr) - 1)):
#             delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
#             gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
#             gae = tf.squeeze(gae)
#             # Corrected tensor update to match scalar shape
#             advantages = tf.tensor_scatter_nd_update(advantages, [[t]], [gae])

#         # Perform multiple epochs of training
#         for _ in range(self.n_epochs):
#             with tf.GradientTape() as tape:
#                 action_probs, values_pred = self.actor_critic(states)
#                 dist = tfp.distributions.Categorical(action_probs)
#                 new_log_probs = dist.log_prob(actions)
#                 new_log_probs = tf.reduce_sum(new_log_probs, axis=1)

#                 ratios = tf.exp(new_log_probs - old_probs)
#                 surr1 = ratios * advantages
#                 surr2 = tf.clip_by_value(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
#                 entropy = tf.reduce_mean(dist.entropy())
#                 actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - self.entropy_coef * entropy
#                 critic_loss = 0.5 * tf.keras.losses.MSE(values_pred, advantages + values)
#                 loss = actor_loss + critic_loss
#             grads = tape.gradient(loss, self.actor_critic.trainable_variables)
#             self.actor_critic.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

#     def train(self):
#         state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = self.memory.get_transitions()
#         self.learn(state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr)
#         self.memory.clear_memory()
    
#     def log_episode(self, stats):
#         """Log statistics."""
#         with self.logger.as_default():
#             for key, value in stats.items():
#                 tf.summary.scalar(key, value, step=self.episodes)
#                 self.logger.flush()
#         self.episodes += 1
