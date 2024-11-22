from factory_sim.multi_factory_env_MA import MultiFactoryEnv
from PPO import Agent
from datetime import datetime
from collections import deque
from IPPO_Config import *

if __name__ == '__main__':
    env = MultiFactoryEnv()
    # input_dims = [len(n) for n in env.reset()]
    # n_agents = len(input_dims)
    n_actions = 5
    agents = [Agent(n_actions=5,
                  action_dims=8,
                  name=name,
                  verbose=verbose,
                  gamma=gamma,
                  lr=lr,
                  gae_lambda=gae_lambda,
                  policy_clip=policy_clip,
                  entropy_coef=entropy_coef,
                  critic_hidden_sizes=critic_hidden_sizes,
                  actor_hidden_sizes=actor_hidden_sizes,
                  batch_size=batch_size,
                  n_epochs=n_epochs) for _ in range(8)]
    
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
            actions, log_probs, val = [], [], []
            for i, obs in enumerate(observation):
                a,l,v = agents[i].choose_action(obs)
                actions.extend(a)
                log_probs.append(l)
                val.append(v)
            observation_, reward, done, info = env.step(actions)
            revenue, energy_cost, storage_cost = info
            reward = reward[0]
            done = done[0]
            total_reward += reward
            total_revenue += sum(revenue)
            total_energy_cost += sum(energy_cost)
            total_storage_cost += sum(storage_cost)
            for i, agent in enumerate(agents):
                agent.store_transition(
                    observation[i],
                    [actions[i]],
                    log_probs[i],
                    val[i],
                    reward,
                    done
                )
                if agent.memory.ready():
                    if not learning_started:
                        learning_started = True
                        print("Beginning Optimisation")
                    agent.train()
            observation = observation_

        ep_rewards.append(total_reward)
        episode_reward_mean = sum(ep_rewards)/len(ep_rewards)
        episode_reward_max = max(ep_rewards)
        # stats = {
        #     "episode_reward":        total_reward,
        #     "episode_reward_mean":   episode_reward_mean,
        #     "episode_reward_max":    episode_reward_max,
        #     "revenue":               total_revenue,
        #     "energy_cost":           total_energy_cost,
        #     "storage_cost":          total_storage_cost,
        # }
        # agent.log_episode(stats)
        print(f"Episode {e}, score: {total_reward}, avg_score: {episode_reward_mean}, estimated completion: {(datetime.now() + ((datetime.now()-now)*(n_episodes-e+1))).strftime('%d %b %y %H:%M:%S')}")
