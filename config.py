class Config:
    map_name = "3m"
    total_steps = 100000
    batch_size = 32
    buffer_size = 5000
    target_update_interval = 200

    # Network Hyperparameters
    rnn_hidden_dim = 64
    lr = 5e-4
    gamma = 0.99

    # Exploration Schedule
    epsilon_start = 1.0
    epsilon_min = 0.05
    epsilon_anneal_steps = 50000
    log_interval = 50