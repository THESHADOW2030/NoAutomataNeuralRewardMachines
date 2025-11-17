from train_grounder import Args


train_grounder_base = Args(

    # General parameters
    model_name = "grounder",
    log_interval = 1,
    save_interval = 10,
    seed = 1,

    # Grounder parameters
    sym_grounder_model = "ObjectCNN",
    obs_size = (56,56),

    # Environment parameters
    max_num_steps = 50,
    env = "GridWorld-v1",
    ltl_sampler = "Dataset_e54",
    progression_mode = "full",

    # Training parameters
    updates = 10000,
    episodes_per_update = 1,
    buffer_size = 2048,
    buffer_start = 512,
    lr = 0.001,
    batch_size = 16,
    update_steps = 64,
    accumulation = 4,

    # Early Stopping
    use_early_stopping = False,
    patience = 20,
    min_delta = 0.0,

    # Evaluation parameters
    evaluate_steps = 256,

    # Agent parameters
    use_agent = False,
    agent_dir = None,
    agent_prob = 0.1,

)