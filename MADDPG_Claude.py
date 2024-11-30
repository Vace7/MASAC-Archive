import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Actor(Model):
    def __init__(self, obs_size, act_size, name="Actor"):
        super().__init__(name=name)
        
        self.l1 = Dense(64, activation='relu', name="L1")
        self.l2 = Dense(64, activation='relu', name="L2")
        self.l3 = Dense(act_size, name="L3", activation='sigmoid')
    
    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        actions = self.l3(x)
        return actions

class Critic(Model):
    def __init__(self, name="Critic"):
        super().__init__(name=name)
        
        self.l1 = Dense(256, activation='relu', name="L1")
        self.l2 = Dense(128, activation='relu', name="L2")
        self.l3 = Dense(1, name="L3")
    
    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        q_value = self.l3(x)
        return q_value

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.pointer = 0
    
    def add(self, obs, act, rew, next_obs, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pointer] = (obs, act, rew, next_obs, done)
        self.pointer = (self.pointer + 1) % self.size
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        obs, act, rew, next_obs, done = zip(*[self.buffer[i] for i in batch])
        return (np.array(obs, dtype=np.float32),
                np.array(act, dtype=np.float32),
                np.array(rew, dtype=np.float32),
                np.array(next_obs, dtype=np.float32),
                np.array(done, dtype=np.float32))
    
    def make_index(self, batch_size):
        return np.random.choice(len(self.buffer), batch_size)
    
    def sample_index(self, idx):
        obs, act, rew, next_obs, done = zip(*[self.buffer[i] for i in idx])
        return (np.array(obs, dtype=np.float32),
                np.array(act, dtype=np.float32),
                np.array(rew, dtype=np.float32),
                np.array(next_obs, dtype=np.float32),
                np.array(done, dtype=np.float32))

class MADDPGAgent:
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, config):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.config = config
        
        # Dimensions
        self.obs_size = obs_shape_n[agent_index]
        self.act_size = act_space_n[agent_index].n
        self.joint_obs_size = sum(obs_shape_n)
        self.joint_act_size = sum(act.n for act in act_space_n)
        
        # Networks
        self.actor = Actor(self.obs_size, self.act_size)
        self.critic = Critic()
        self.target_actor = Actor(self.obs_size, self.act_size)
        self.target_critic = Critic()
        
        # Copy weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Optimizers
        self.actor_optimizer = Adam(learning_rate=config.actor_lr)
        self.critic_optimizer = Adam(learning_rate=config.critic_lr)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = config.batch_size * config.max_episode_len
        
        # Other parameters
        self.gamma = config.gamma
        self.tau = config.tau
        
    @tf.function
    def get_action(self, obs):
        return self.actor(obs)
    
    def action(self, obs):
        obs_tensor = tf.convert_to_tensor([obs], dtype=tf.float32)
        action = self.get_action(obs_tensor)
        return action[0].numpy()
    
    def experience(self, obs, act, rew, next_obs, done, terminal):
        self.replay_buffer.add(obs, act, rew, next_obs, float(done))
    
    @tf.function
    def _update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(self.tau * b + (1 - self.tau) * a)
    
    def update(self, agents, t):
        if len(self.replay_buffer.buffer) < self.max_replay_buffer_len:
            return
        
        if t % 4 != 0:  # only update every 100 steps
            return
        
        # Sample batch
        indices = self.replay_buffer.make_index(self.config.batch_size)
        
        # Collect experiences from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        
        for i in range(self.n):
            obs, act, _, obs_next, _ = agents[i].replay_buffer.sample_index(indices)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(indices)
        rew = np.expand_dims(rew, axis=-1)
        done = np.expand_dims(done, axis=-1)
        
        # Update critic
        with tf.GradientTape() as tape:
            # Target actions
            target_acts_next = [agent.target_actor(next_obs) 
                              for agent, next_obs in zip(agents, obs_next_n)]
            
            # Q-value for next state
            target_q_input = tf.concat([tf.concat(obs_next_n, axis=-1),
                                      tf.concat(target_acts_next, axis=-1)], axis=-1)
            target_q = self.target_critic(target_q_input)
            
            # Q-target
            q_target = rew + self.gamma * (1.0 - done) * target_q
            
            # Q-value for current state
            q_input = tf.concat([tf.concat(obs_n, axis=-1),
                               tf.concat(act_n, axis=-1)], axis=-1)
            q_value = self.critic(q_input)
            
            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(q_value - q_target))
        
        # Update critic
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            # Actor output
            new_actions = self.actor(obs)
            
            # Change only this agent's actions
            act_n[self.agent_index] = new_actions
            
            # Q-value for updated actions
            q_input = tf.concat([tf.concat(obs_n, axis=-1),
                               tf.concat(act_n, axis=-1)], axis=-1)
            actor_loss = -tf.reduce_mean(self.critic(q_input))
        
        # Update actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))
        
        # Update target networks
        self._update_target(self.target_actor.variables, self.actor.variables)
        self._update_target(self.target_critic.variables, self.critic.variables)
        
        return {
            'critic_loss': critic_loss.numpy(),
            'actor_loss': actor_loss.numpy(),
            'target_q_mean': tf.reduce_mean(target_q).numpy(),
            'reward_mean': tf.reduce_mean(rew).numpy(),
            'target_q_std': tf.std(target_q).numpy()
        }
    
    def save(self, path):
        self.actor.save_weights(f"{path}/actor_{self.agent_index}")
        self.critic.save_weights(f"{path}/critic_{self.agent_index}")
        self.target_actor.save_weights(f"{path}/target_actor_{self.agent_index}")
        self.target_critic.save_weights(f"{path}/target_critic_{self.agent_index}")
    
    def load(self, path):
        self.actor.load_weights(f"{path}/actor_{self.agent_index}")
        self.critic.load_weights(f"{path}/critic_{self.agent_index}")
        self.target_actor.load_weights(f"{path}/target_actor_{self.agent_index}")
        self.target_critic.load_weights(f"{path}/target_critic_{self.agent_index}")