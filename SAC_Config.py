# Training/Testing configuration
load_checkpoint = False

if load_checkpoint:
    n_episodes = 1
    verbose = False
    deterministic = True
else:
    n_episodes = 100
    verbose = True
    deterministic = False
    
# SAC HYPER PARAMETERS #
alpha = 0.2                       # Temperature parameter for the entropy
polyak = 0.995                  # Polyak parameter for the weights copy
lr = 1e-3                        # Learning rate for the networks
gamma = 0.99                    # Discount factor
autotune = True                 # Select autotuning of alpha
utd_ratio = 20                  # Number of times to update critic network per timestep
use_layer_norm = True
name = f"SAC-Continuous-{lr}"

# MLP PARAMETERS #
critic_hidden_sizes = (256,256)          # Number of the hidden layers for the critics
actor_hidden_sizes = (256,256)

# BUFFER PARAMETERS #
batch_size = 256                        # Get batch_size sample from the ReplayBuffer
max_size = int(500*n_episodes)                  # ReplayBuffer size
