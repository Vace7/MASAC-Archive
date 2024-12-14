from factory_sim.multi_factory_env_MA import MultiFactoryEnv
from DDQN import DDQNAgent, ReplayBuffer
from DDQN_Config import *
from datetime import datetime
from collections import deque
import csv

if __name__ == '__main__':
    env = MultiFactoryEnv()
    input_dims = [len(obs) for obs in env.reset()]
    n_agents = 8
    n_actions = 5
    epsilon = 1
    agent = DDQNAgent(
        n_agents=n_agents, 
        obs_dims=input_dims, 
        n_actions=n_actions,
        lr=lr, 
        gamma=gamma,
        polyak=polyak,
        hidden_sizes=hidden_sizes,
        verbose=verbose, 
        name=name)

    buffer = ReplayBuffer(
        obs_dims=len(input_dims),
        buffer_size=max_size,
        n_agents=n_agents,
        batch_size=batch_size,
        n_actions=n_actions)
    
    ep_rewards = deque(maxlen=20)
    learning_started = False

    with open("../tracking/"+name+"_"+datetime.now().strftime('%Y%m%d-%H%M%S')+'.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(('Step', 'Reward', 'Revenue', 'Energy Cost', 'Storage Cost'))
        for e in range(n_episodes):
            epsilon*=epsilon_decay
            observation = env.reset()
            done = False
            total_reward = 0
            total_revenue = 0
            total_energy_cost = 0
            total_storage_cost = 0
            now = datetime.now()
            while not done:
                actions = agent.select_action(observation, epsilon)
                observation_, reward, done, info = env.step(actions)
                revenue, energy_cost, storage_cost = info
                total_reward += reward[0]
                total_revenue += sum(revenue)
                total_energy_cost += sum(energy_cost)
                total_storage_cost += sum(storage_cost)
                buffer.store_transition(
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
                    agent.train(buffer)
                observation = observation_
                done = done[0]

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
                        "storage_cost":          total_storage_cost
                    }
                    agent.log_episode(stats)
            if load_checkpoint:
                print(f"Evaluation Score: {total_reward}")
            else:
                print(f"Episode {e}, score: {total_reward}, avg_score: {episode_reward_mean}, estimated completion: {(datetime.now() + ((datetime.now()-now)*(n_episodes-e+1))).strftime('%d %b %y %H:%M:%S')}")
