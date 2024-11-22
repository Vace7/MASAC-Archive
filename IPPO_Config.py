# Training/Testing configuration
load_checkpoint = False

if load_checkpoint:
    n_episodes = 1
    verbose = False
else:
    n_episodes = 100
    verbose = False
    
# PPO HYPER PARAMETERS #
gae_lambda = 0.95
policy_clip = 0.2
entropy_coef = 0.001
lr = 3e-4                        # Learning rate for the networks
gamma = 0.99                    # Discount factor
name = f"IPPO-{lr}"
n_epochs = 6

# MLP PARAMETERS #
critic_hidden_sizes = (256,256)          # Number of the hidden layers for the critics
actor_hidden_sizes = (256,256)

# BUFFER PARAMETERS #
batch_size = 500
