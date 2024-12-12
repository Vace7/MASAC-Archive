# Training/Testing configuration
load_checkpoint = False

if load_checkpoint:
    n_episodes = 1
    verbose = False
    deterministic = True
else:
    n_episodes = 300
    verbose = True
    deterministic = False
    
# SAC HYPER PARAMETERS #
alpha = 0.2                       # Temperature parameter for the entropy
polyak = 0.995                  # Polyak parameter for the weights copy
lr = 3e-4                        # Learning rate for the networks
gamma = 0.99                    # Discount factor
autotune = True                 # Select autotuning of alpha
utd_ratio = 1                  # Number of times to update critic network per timestep
use_layer_norm = False
name = f"MASAC-Discrete-{lr}-layernorm-AdamW"

# MLP PARAMETERS #
critic_hidden_sizes = (256,128)          # Number of the hidden layers for the critics
actor_hidden_sizes = (64,64)

# BUFFER PARAMETERS #
batch_size = 256                        # Get batch_size sample from the ReplayBuffer
learning_starts = 5
max_size = int(500*n_episodes)                  # ReplayBuffer size
