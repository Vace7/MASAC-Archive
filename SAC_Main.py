import numpy as np
from factory_sim.multi_factory_env_SA import MultiFactoryEnv
from SAC_Config import *
from SAC import SACAgent
from SAC import ReplayBuffer
from collections import deque
from datetime import datetime

def map_continuous_to_discrete_array(actions, n):
        """
        Maps a NumPy array of continuous actions in the range [-1, 1] to a discrete space of values [0, 1, 2, ..., n-1].

        Parameters:
        - actions: np.ndarray, array of continuous actions in the range [-1, 1]
        - n: int, number of discrete actions

        Returns:
        - np.ndarray, array of discrete actions in the range [0, n-1]
        """
        actions = np.clip(actions, -1, 1)
        discrete_actions = (actions + 1) * (n - 1) / 2
        discrete_actions = np.round(discrete_actions).astype(int)
        return discrete_actions

if __name__ == '__main__':
    env = MultiFactoryEnv()
    input_dims=22
    n_actions=8
    agent = SACAgent(
        input_dims=input_dims, 
        action_dims=n_actions, 
        action_bound=1,
        alpha=alpha,
        polyak=polyak,
        lr=lr,
        gamma=gamma,
        autotune=autotune,
        critic_hidden_sizes=critic_hidden_sizes,
        actor_hidden_sizes=actor_hidden_sizes,
        use_layer_norm=use_layer_norm,
        utd_ratio=utd_ratio,
        name=name,
        verbose=verbose)
    
    # agent = SACAgent(
    #     input_dims=input_dims, 
    #     action_dims=n_actions, 
    #     action_bound=1,
    #     alpha=alpha,
    #     polyak=polyak,
    #     lr=lr,
    #     gamma=gamma,
    #     autotune=autotune,
    #     hidden_sizes=critic_hidden_sizes,
    #     use_layer_norm=use_layer_norm,
    #     utd_ratio=utd_ratio,
    #     name=name,
    #     verbose=verbose)
    
    buffer = ReplayBuffer(
        obs_dim=input_dims,
        act_dim=n_actions,
        size=max_size,
        batch_size=batch_size)

    ep_rewards = deque(maxlen=20)
    learning_started = False

    for i in range(n_episodes):
        observation = env.reset()
        observation = observation.astype(np.float32)
        done = False
        score = 0
        total_revenue = 0
        total_energy_cost = 0
        total_storage_cost = 0
        now = datetime.now()
        while not done:
            action = agent.select_action(observation, deterministic=deterministic)
            observation_, reward, done, info = env.step(map_continuous_to_discrete_array(action, 5))
            revenue, energy_cost, storage_cost = info
            observation_ = observation_.astype(np.float32)
            reward = reward[0]
            done = done[0]
            score += reward
            total_revenue += sum(revenue)
            total_energy_cost += sum(energy_cost)
            total_storage_cost += sum(storage_cost)
            buffer.store(
                observation, 
                action, 
                reward, 
                observation_, 
                done)

            if not load_checkpoint and buffer.ready():
                if not learning_started:
                    learning_started = True
                    print("Buffer ready, beginning optimisation.")
                pi_loss, q_loss, trained_alpha = agent.train(buffer.sample_batch())
            observation = observation_

        ep_rewards.append(score)
        episode_reward_mean = sum(ep_rewards)/len(ep_rewards)
        episode_reward_max = max(ep_rewards)
        if not load_checkpoint and learning_started:
            if verbose:
                stats = {
                    "episode_reward":        score,
                    "episode_reward_mean":   episode_reward_mean,
                    "episode_reward_max":    episode_reward_max,
                    "revenue":               total_revenue,
                    "energy_cost":           total_energy_cost,
                    "storage_cost":          total_storage_cost,
                    "actor_losses" :         pi_loss,
                    "critic_losses" :        q_loss,
                    "alpha" :                trained_alpha
                    }
                agent.log_episode(stats)
        if load_checkpoint:
            print(f"Evaluation Score: {score}")
        else:
            print(f"Episode {i}, score: {score}, avg_score: {episode_reward_mean}, estimated completion: {(datetime.now() + ((datetime.now()-now)*(n_episodes-i+1))).strftime('%d %b %y %H:%M:%S')}")


