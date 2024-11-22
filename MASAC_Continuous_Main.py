import numpy as np
from factory_sim.multi_factory_env_MA import MultiFactoryEnv
# from MASAC_not_working import MASACAgent, MASACReplayBuffer  # Not working due to lack of accounting for inter-agent dependencies
# from MASAC_claude_1 import MASACAgent, MASACReplayBuffer  # Accounts for inter-agent action dependencies
# from MASAC_claude_2 import MASACAgent, MASACReplayBuffer  # Improves by using multi-headed q structure for fast computation
from MASAC_Continuous import MASACAgent, MASACReplayBuffer  # Improves by using multiple alphas to vary exploration between actors
# from MASAC_multicritic import MASACAgent, MASACReplayBuffer  # Uses multiple critics, not very effective
from MASAC_Continuous_Config import *
from datetime import datetime
from collections import deque

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
    input_dims = [len(n) for n in env.reset()]
    n_agents = len(input_dims)
    n_actions=1
    agent = MASACAgent(
        n_agents=n_agents, 
        obs_dims=sum(input_dims), 
        act_dims=n_actions,
        alpha=alpha,
        polyak=polyak,
        lr=lr, 
        gamma=gamma,
        autotune=autotune,
        critic_hidden_sizes=critic_hidden_sizes,
        actor_hidden_sizes=actor_hidden_sizes,
        use_layer_norm=use_layer_norm,
        utd_ratio=utd_ratio,
        verbose=verbose, 
        name=name)

    buffer = MASACReplayBuffer(
        obs_dims=input_dims,
        act_dims=1,
        size=max_size,
        n_agents=n_agents,
        batch_size=batch_size)
    
    ep_rewards = deque(maxlen=20)
    learning_started = False

    for e in range(n_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        total_revenue = 0
        total_energy_cost = 0
        total_storage_cost = 0
        now = datetime.now()
        while not done:
            actions = agent.select_action(observation, deterministic)
            observation_, reward, done, info = env.step(map_continuous_to_discrete_array(np.array(actions), 5).flatten())
            revenue, energy_cost, storage_cost = info
            reward = reward[0]
            done = done[0]
            total_reward += reward
            total_revenue += sum(revenue)
            total_energy_cost += sum(energy_cost)
            total_storage_cost += sum(storage_cost)
            buffer.store(
                observation,
                actions,
                reward,
                observation_,
                done
            )
            if not load_checkpoint and buffer.ready():
                if not learning_started:
                    learning_started = True
                    print("Buffer ready, beginning optimisation.")
                pi_losses, q_losses, alpha = agent.train(buffer.sample_batch())
            observation = observation_

        ep_rewards.append(total_reward)
        episode_reward_mean = sum(ep_rewards)/len(ep_rewards)
        episode_reward_max = max(ep_rewards)
        if not load_checkpoint and learning_started:
            stats = {
                "episode_reward":        total_reward,
                "episode_reward_mean":   episode_reward_mean,
                "episode_reward_max":    episode_reward_max,
                "revenue":               total_revenue,
                "energy_cost":           total_energy_cost,
                "storage_cost":          total_storage_cost,
                "critic_losses":         q_losses,
                # "actor_losses":          sum(pi_losses)/len(pi_losses),
                # "alpha":                 sum(alpha)/len(alpha)
                "actor_losses":          pi_losses,
                "alpha":                 alpha
            }
            agent.log_episode(stats)
        if load_checkpoint:
            print(f"Evaluation Score: {total_reward}")
        else:
            print(f"Episode {e}, score: {total_reward}, avg_score: {episode_reward_mean}, estimated completion: {(datetime.now() + ((datetime.now()-now)*(n_episodes-e+1))).strftime('%d %b %y %H:%M:%S')}")
