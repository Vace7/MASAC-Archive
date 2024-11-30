# Training/Testing configuration
load_checkpoint = False

if load_checkpoint:
    n_episodes = 1
    verbose = False
else:
    n_episodes = 300
    verbose = True
    
# HYPER PARAMETERS #
polyak=1
lr = 1e-4                        # Learning rate for the networks
gamma = 0.95                    # Discount factor
epsilon_decay = 0.988
name = f"DDQN-{lr}"

# MLP PARAMETERS #
hidden_sizes = (256,256)          # Number of the hidden layers for the critics

# BUFFER PARAMETERS #
batch_size = 256
max_size = n_episodes*500
