# Training parameters
n_episodes = 300
batch_size = 256
max_size = 500*n_episodes

# Algorithm parameters
gamma = 0.95
tau = 0.01
lr=1e-3
epsilon_decay = 0.95

# Logging
verbose = True
name = f"MADDPG-{lr}-conservative"