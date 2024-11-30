from factory_sim.multi_factory_env_MA import MultiFactoryEnv
from datetime import datetime
from collections import deque
from MADDPG_Claude import MADDPGAgent
import numpy as np
import tensorflow as tf
import os

class Config:
    def __init__(self):
        # Training parameters
        self.n_episodes = 300
        self.max_steps = 500
        self.batch_size = 1000
        self.max_episode_len = 500
        self.learning_starts = 1000
        
        # Agent parameters
        self.actor_lr = 0.01
        self.critic_lr = 0.01
        self.gamma = 0.95
        self.tau = 0.01
        
        # Exploration parameters
        self.exploration_noise = 0.1
        self.noise_decay = 0.995
        
        # Logging parameters
        self.log_interval = 100
        self.checkpoint_interval = 1000
        self.save_dir = "checkpoints"
        
        # Environment parameters
        self.n_actions = 5  # Number of discrete actions

def train_maddpg(config):
    # Environment setup
    env = MultiFactoryEnv()
    observations = env.reset()
    
    # Get environment dimensions
    input_dims = [len(obs) for obs in observations]
    n_agents = len(input_dims)
    
    # Initialize agents
    agents = []
    for i in range(n_agents):
        agent_config = Config()
        agents.append(MADDPGAgent(
            name=f"agent_{i}",
            obs_shape_n=input_dims,
            act_space_n=[type('ActionSpace', (), {'n': config.n_actions})] * n_agents,
            agent_index=i,
            config=agent_config
        ))
    
    # Training metrics
    ep_rewards = deque(maxlen=20)
    best_reward = float('-inf')
    noise_scale = config.exploration_noise
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Training loop
    for episode in range(config.n_episodes):
        observations = env.reset()
        episode_reward = 0
        episode_metrics = {
            'revenue': 0,
            'energy_cost': 0,
            'storage_cost': 0
        }
        
        start_time = datetime.now()
        
        for step in range(config.max_steps):
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents):
                action = agent.action(observations[i])
                # Add exploration noise
                action += np.random.normal(0, noise_scale, size=action.shape)
                action = tf.argmax(action).numpy()
                actions.append(action)

            # Environment step
            next_observations, rewards, dones, info = env.step(actions)
            revenue, energy_cost, storage_cost = info
            # Update metrics
            episode_reward += rewards[0]
            episode_metrics['revenue'] += sum(revenue)
            episode_metrics['energy_cost'] += sum(energy_cost)
            episode_metrics['storage_cost'] += sum(storage_cost)
            
            # Store experience in all agents
            for i, agent in enumerate(agents):
                agent.experience(
                    observations[i],
                    actions[i],
                    rewards[i],
                    next_observations[i],
                    dones[i],
                    dones[i]
                )
            
            # Update all agents
            if episode >= config.learning_starts:
                for agent in agents:
                    update_info = agent.update(agents, step)
                    if update_info:
                        episode_metrics.update(update_info)
            
            observations = next_observations
            
            if any(dones):
                break
        
        # Episode complete
        ep_rewards.append(episode_reward)
        avg_reward = sum(ep_rewards) / len(ep_rewards)
        
        # Decay exploration noise
        noise_scale *= config.noise_decay
        
        duration = datetime.now() - start_time
        estimated_completion = datetime.now() + (duration * (config.n_episodes - episode))
        
        print(f"\nEpisode {episode}/{config.n_episodes}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Revenue: {episode_metrics['revenue']:.2f}")
        print(f"Energy Cost: {episode_metrics['energy_cost']:.2f}")
        print(f"Storage Cost: {episode_metrics['storage_cost']:.2f}")
        print(f"Exploration Noise: {noise_scale:.3f}")
        print(f"Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'critic_loss' in episode_metrics:
            print(f"Critic Loss: {episode_metrics['critic_loss']:.4f}")
            print(f"Actor Loss: {episode_metrics['actor_loss']:.4f}")
        
        # Save checkpoints
        # if episode % config.checkpoint_interval == 0 or episode_reward > best_reward:
        #     checkpoint_path = os.path.join(config.save_dir, f"episode_{episode}")
        #     os.makedirs(checkpoint_path, exist_ok=True)
            
            # for agent in agents:
            #     agent.save(checkpoint_path)
            
            # if episode_reward > best_reward:
            #     best_reward = episode_reward
            #     best_path = os.path.join(config.save_dir, "best")
            #     os.makedirs(best_path, exist_ok=True)
            #     for agent in agents:
            #         agent.save(best_path)
    
    return agents

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create and load config
    config = Config()
    
    # Train agents
    trained_agents = train_maddpg(config)
    
    # # Save final model
    # final_path = os.path.join(config.save_dir, "final")
    # os.makedirs(final_path, exist_ok=True)
    # for agent in trained_agents:
    #     agent.save(final_path)