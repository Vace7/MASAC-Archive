from factory_sim.multi_factory_env_MA import MultiFactoryEnv
from MADDPG_new import MADDPGAgent, ReplayBuffer
from MADDPG_new_Config import *
from datetime import datetime
from collections import deque

if __name__ == '__main__':
    # Initialize environment (assuming similar to your factory environment)
    env = MultiFactoryEnv()  # Replace with your environment
    observation = env.reset()
    
    # Get dimensions
    input_dims = [len(obs) for obs in observation]
    n_agents = len(input_dims)
    # Initialize agent
    agent = MADDPGAgent(
        obs_dim=input_dims,
        num_actions=5,
        num_agents=n_agents,
        lr=lr,
        gamma=gamma,
        tau=tau,
        name=name,
        verbose=verbose
    )
    for _ in range(20):
        agent.reset_weights()
        epsilon = 1
        
        # Initialize replay buffer
        buffer = ReplayBuffer(
            buffer_size=max_size,
            batch_size=batch_size
        )
        
        # Training metrics
        ep_rewards = deque(maxlen=20)
        learning_started = False
        
        # Training loop
        for e in range(n_episodes):
            observation = env.reset()
            done = [False,False]
            total_reward = 0
            total_revenue = 0
            total_energy_cost = 0
            total_storage_cost = 0
            now = datetime.now()
            epsilon = epsilon*epsilon_decay
            while not any(done):
                # Get actions
                actions = []
                actions_record = []
                for agent_idx in range(n_agents):
                    # Select actions for each agent
                    probs, action = agent.select_action(observation[agent_idx], agent_idx, epsilon=epsilon)
                    actions.append(action)
                    actions_record.append(probs)
                
                # Environment step
                observation_, reward, done, info = env.step(actions)
                revenue, energy_cost, storage_cost = info
                
                # Update metrics
                total_reward += reward[0]
                total_revenue += sum(revenue)
                total_energy_cost += sum(energy_cost)
                total_storage_cost += sum(storage_cost)
                buffer.store_transition(
                    observation,
                    actions_record,
                    reward,
                    observation_,
                    done
                )
                
                agent.train(buffer)     
                observation = observation_
            
            # Episode complete - update statistics
            ep_rewards.append(total_reward)
            episode_reward_mean = sum(ep_rewards) / len(ep_rewards)
            episode_reward_max = max(ep_rewards)
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
            # Print progress

            print(f"Episode {e}, score: {total_reward}, avg_score: {episode_reward_mean}, estimated completion: {(datetime.now() + ((datetime.now()-now)*(n_episodes-e+1))).strftime('%d %b %y %H:%M:%S')}")