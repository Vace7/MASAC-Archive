# Training parameters
n_episodes = 300
batch_size = 1000
max_size = 500*n_episodes
deterministic = False
load_checkpoint = False

# Algorithm parameters
gamma = 0.95
polyak = 0.99
lr=1e-3
lr_actor = lr
lr_critic = lr
critic_hidden_sizes = (256, 128)
actor_hidden_sizes = (64, 64)
learning_starts = 5

# Logging
verbose = True
name = f"MADDPG-{lr}"