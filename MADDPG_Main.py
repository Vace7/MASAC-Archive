# from factory_sim.multi_factory_env_MA import MultiFactoryEnv
# # from MADDPG import MADDPGAgent, MADDPGReplayBuffer
# from MADDPG import ContinuousMADDPGAgent, MADDPGReplayBuffer
# from MADDPG_Config import *
# from datetime import datetime
# from collections import deque
# import numpy as np
# import random
# import tensorflow as tf

# def convert_to_onehot(actions: list, n_actions: int) -> list:
#     """Convert discrete actions to one-hot vectors"""
#     return [tf.one_hot(a, n_actions).numpy() for a in actions]

# if __name__ == '__main__':
#     # Environment setup
#     env = MultiFactoryEnv()
#     observations = env.reset()
#     input_dims = [len(obs) for obs in observations]
#     n_agents = len(input_dims)
#     n_actions = 5
    
#     # Agent initialization
#     agent = MADDPGAgent(
#         n_agents=n_agents,
#         obs_dims=input_dims,
#         n_actions=n_actions,  # Same action space for all agents
#         gamma=gamma,
#         polyak=polyak,
#         lr_actor=lr_actor,
#         lr_critic=lr_critic,
#         critic_hidden_sizes=critic_hidden_sizes,
#         actor_hidden_sizes=actor_hidden_sizes,
#         verbose=verbose,
#         name=name
#     )

#     # Buffer initialization
#     buffer = MADDPGReplayBuffer(
#         n_agents=n_agents,
#         obs_dims=input_dims,
#         action_dims=[n_actions]*n_agents,
#         buffer_size=max_size,
#         batch_size=batch_size
#     )
    
#     # Training setup
#     ep_rewards = deque(maxlen=20)
#     learning_started = False

#     for e in range(n_episodes):
#         observations = env.reset()
#         done = False
#         total_reward = 0
#         total_revenue = 0
#         total_energy_cost = 0
#         total_storage_cost = 0
#         episode_actor_losses = []
#         episode_critic_losses = []
#         now = datetime.now()

#         while not done:
#             # Action selection
#             if learning_started:
#                 actions_probs = [
#                     agent.actors[i](
#                         tf.convert_to_tensor([observations[i]], dtype=tf.float32)
#                     ).numpy()[0]
#                     for i in range(n_agents)
#                 ]
#                 if deterministic:
#                     actions = [np.argmax(probs) for probs in actions_probs]
#                 else:
#                     actions = [
#                         np.random.choice(n_actions, p=probs)
#                         for probs in actions_probs
#                     ]
#             else:
#                 actions = [random.randint(0, n_actions-1) for _ in range(n_agents)]

#             # Environment step
#             next_observations, rewards, dones, info = env.step(actions)
#             revenue, energy_cost, storage_cost = info
#             reward = rewards[0]  # Using first agent's reward as total reward
#             done = dones[0]  # Using first agent's done signal
            
#             # Update statistics
#             total_reward += reward
#             total_revenue += sum(revenue)
#             total_energy_cost += sum(energy_cost)
#             total_storage_cost += sum(storage_cost)

#             # Store experience
#             buffer.store(
#                 observations,
#                 convert_to_onehot(actions, n_actions),
#                 rewards,
#                 next_observations,
#                 dones
#             )

#             # Training
#             if not load_checkpoint and buffer.ready():
#                 if e >= learning_starts:
#                     if not learning_started:
#                         learning_started = True
#                         print("Buffer ready, beginning optimization.")
           
#                     obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = buffer.sample()
                    
#                     # Update all agents
#                     agent.train_step(
#                         obs_batch,
#                         action_batch,
#                         reward_batch,
#                         next_obs_batch,
#                         done_batch
#                     )

#             observations = next_observations

#         # Episode end updates
#         ep_rewards.append(total_reward)
#         episode_reward_mean = sum(ep_rewards)/len(ep_rewards)
#         episode_reward_max = max(ep_rewards)

#         # Logging
#         if not load_checkpoint and verbose:
#             stats = {
#                 "episode_reward": total_reward,
#                 "episode_reward_mean": episode_reward_mean,
#                 "episode_reward_max": episode_reward_max,
#                 "revenue": total_revenue,
#                 "energy_cost": total_energy_cost,
#                 "storage_cost": total_storage_cost,
#             }
#             agent.log_episode(stats)

#         # Progress output
#         if load_checkpoint:
#             print(f"Evaluation Score: {total_reward}")
#         else:
#             estimated_completion = datetime.now() + ((datetime.now()-now)*(n_episodes-e+1))
#             print(
#                 f"Episode {e}, "
#                 f"score: {total_reward:.2f}, "
#                 f"avg_score: {episode_reward_mean:.2f}, "
#                 f"estimated completion: {estimated_completion.strftime('%d %b %y %H:%M:%S')}"
#             )

from factory_sim.multi_factory_env_MA import MultiFactoryEnv
from MADDPG import ContinuousMADDPGAgent, MADDPGReplayBuffer
from MADDPG_Config import *
from datetime import datetime
from collections import deque
import numpy as np
def map_continuous_to_discrete(actions: np.ndarray, n: int) -> np.ndarray:
    """
    Maps continuous actions in [-1, 1] to discrete actions [0, n-1]
    
    Args:
        actions: Continuous actions in range [-1, 1]
        n: Number of discrete actions
    
    Returns:
        Discrete actions in range [0, n-1]
    """
    actions = np.clip(actions, -1, 1)
    discrete_actions = (actions + 1) * (n - 1) / 2
    return np.round(discrete_actions).astype(int)


if __name__ == '__main__':
    # Initialize environment (assuming similar to your factory environment)
    env = MultiFactoryEnv()  # Replace with your environment
    observation = env.reset()
    
    # Get dimensions
    input_dims = [len(obs) for obs in observation]
    n_agents = len(input_dims)
    action_dims = [1 for _ in range(n_agents)]  # Assuming 1D continuous actions
    action_high = 1.0
    
    # Initialize agent
    agent = ContinuousMADDPGAgent(
        n_agents=n_agents,
        obs_dims=input_dims,
        action_dims=action_dims,
        action_high=action_high,
        gamma=gamma,
        polyak=polyak,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        critic_hidden_sizes=critic_hidden_sizes,
        actor_hidden_sizes=actor_hidden_sizes,
        verbose=verbose,
        name=name
    )
    
    # Initialize replay buffer
    buffer = MADDPGReplayBuffer(
        obs_dims=input_dims,
        act_dims=action_dims,
        size=max_size,
        n_agents=n_agents,
        batch_size=batch_size
    )
    
    # Training metrics
    ep_rewards = deque(maxlen=20)
    learning_started = False
    
    # Training loop
    for e in range(n_episodes):
        observation = env.reset()
        agent.reset_noise()  # Reset exploration noise
        done = False
        total_reward = 0
        total_revenue = 0
        total_energy_cost = 0
        total_storage_cost = 0
        now = datetime.now()
        
        while not done:
            # Get actions
            actions = agent.get_action(observation, add_noise=not deterministic)
            discrete_actions = map_continuous_to_discrete(actions, 5)
            # Environment step
            observation_, reward, done, info = env.step(discrete_actions)
            revenue, energy_cost, storage_cost = info
            reward = reward[0]  # Assuming shared reward
            done = done[0]  # Assuming shared done signal
            
            # Update metrics
            total_reward += reward
            total_revenue += sum(revenue)
            total_energy_cost += sum(energy_cost)
            total_storage_cost += sum(storage_cost)
            
            # Store transition
            buffer.store(
                observation,
                actions,
                reward,
                observation_,
                done
            )
            
            # Train if buffer is ready
            if buffer.ready():
                if not learning_started:
                    learning_started = True
                    print("Buffer ready, beginning optimization.")
                batch = buffer.sample_batch()
                critic_losses, actor_losses = agent.train_step(
                    batch['observations'],
                    batch['actions'],
                    batch['rewards'],
                    batch['next_observations'],
                    batch['dones']
                )
            
            observation = observation_
        
        # Episode complete - update statistics
        ep_rewards.append(total_reward)
        episode_reward_mean = sum(ep_rewards) / len(ep_rewards)
        episode_reward_max = max(ep_rewards)
        
        # Log statistics
        if learning_started:
            stats = {
                "episode_reward": total_reward,
                "episode_reward_mean": episode_reward_mean,
                "episode_reward_max": episode_reward_max,
                "revenue": total_revenue,
                "energy_cost": total_energy_cost,
                "storage_cost": total_storage_cost,
                "critic_losses": sum(critic_losses) / len(critic_losses),
                "actor_losses": sum(actor_losses) / len(actor_losses)
            }
            agent.log_episode(stats)
        
        # Print progress
        estimated_completion = datetime.now() + ((datetime.now() - now) * (n_episodes - e + 1))
        print(f"Episode {e}, "
              f"score: {total_reward:.2f}, "
              f"avg_score: {episode_reward_mean:.2f}, "
              f"estimated completion: {estimated_completion.strftime('%d %b %y %H:%M:%S')}")