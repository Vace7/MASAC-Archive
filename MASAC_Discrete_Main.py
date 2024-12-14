from factory_sim.multi_factory_env_MA import MultiFactoryEnv
from MASAC_Discrete import MASACAgent, MASACReplayBuffer
from MASAC_Discrete_Config import *
from datetime import datetime
from collections import deque
import random
import csv

if __name__ == '__main__':
    env = MultiFactoryEnv()
    input_dims = [len(n) for n in env.reset()]
    n_agents = len(input_dims)
    n_actions = 5
    agent = MASACAgent(
        n_agents=n_agents, 
        obs_dims=sum(input_dims), 
        n_actions=n_actions,
        alpha=alpha,
        polyak=polyak,
        lr=lr, 
        gamma=gamma,
        autotune=autotune,
        critic_hidden_sizes=critic_hidden_sizes,
        actor_hidden_sizes=actor_hidden_sizes,
        utd_ratio=utd_ratio,
        use_layer_norm=use_layer_norm,
        verbose=verbose, 
        name=name)
    
    buffer = MASACReplayBuffer(
        obs_dims=input_dims,
        size=max_size,
        n_agents=n_agents,
        batch_size=batch_size)
    
    ep_rewards = deque(maxlen=20)
    learning_started = False

    with open("../tracking/"+name+"_"+datetime.now().strftime('%Y%m%d-%H%M%S')+'.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(('Step', 'Reward', 'Revenue', 'Energy Cost', 'Storage Cost'))
        for e in range(n_episodes):
            observation = env.reset()
            done = False
            total_reward = 0
            total_revenue = 0
            total_energy_cost = 0
            total_storage_cost = 0
            now = datetime.now()
            while not done:
                if learning_started:
                    actions = agent.select_action(observation, deterministic)
                else:
                    actions = [random.randint(0,4) for _ in range(8)]
                observation_, reward, done, info = env.step(actions)
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
                    pi_losses, q_losses, alpha = 0,0,0
                    if e>=learning_starts:
                        if not learning_started:
                            learning_started = True
                            print("Buffer ready, beginning optimisation.")
                        pi_losses, q_losses, alpha = agent.train(buffer.sample_batch())
                observation = observation_

            ep_rewards.append(total_reward)
            episode_reward_mean = sum(ep_rewards)/len(ep_rewards)
            episode_reward_max = max(ep_rewards)
            writer.writerow((e, total_reward, total_revenue, total_energy_cost, total_storage_cost))
            if not load_checkpoint:
                if verbose:
                    stats = {
                        "episode_reward":        total_reward,
                        "episode_reward_mean":   episode_reward_mean,
                        "episode_reward_max":    episode_reward_max,
                        "revenue":               total_revenue,
                        "energy_cost":           total_energy_cost,
                        "storage_cost":          total_storage_cost,
                        "actor_losses":          pi_losses,
                        "critic_losses":         q_losses,
                        "alpha":                 alpha
                    }
                    agent.log_episode(stats)
            if load_checkpoint:
                print(f"Evaluation Score: {total_reward}")
            else:
                print(f"Episode {e}, score: {total_reward}, avg_score: {episode_reward_mean}, estimated completion: {(datetime.now() + ((datetime.now()-now)*(n_episodes-e+1))).strftime('%d %b %y %H:%M:%S')}")
