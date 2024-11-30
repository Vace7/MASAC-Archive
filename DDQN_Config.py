# Training/Testing configuration
load_checkpoint = False

if load_checkpoint:
    n_episodes = 1
    verbose = False
else:
    n_episodes = 300
    verbose = True
    
# HYPER PARAMETERS #
lr = 1e-4                        # Learning rate for the networks
gamma = 0.95                    # Discount factor
polyak = 0.995
epsilon_decay = 0.95
name = f"DDQN-{lr}"

# MLP PARAMETERS #
hidden_sizes = (64,64)          # Number of the hidden layers for the critics

# BUFFER PARAMETERS #
batch_size = 256
max_size = n_episodes*500
