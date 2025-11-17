from train_agent import Args


test_simple_ltl_5l = Args(

    # General parameters
    model_name = "gnn_pretrain",
    algo = "ppo",
    seed = 1,
    log_interval = 10,
    save_interval = 100,
    procs = 16,
    frames_per_proc = 512,
    frames = 20000000,

    # Environment parameters
    env = "Simple-LTL-Env-5L-v0",
    ltl_sampler = "Eventually_1_5_1_4",

    # GNN parameters
    gnn_model = "RGCN_8x32_ROOT_SHARED",
    use_pretrained_gnn = False,
    gnn_pretrain = None,
    freeze_gnn = False,

    # Grounder parameters
    grounder_model = None,

    # Agent parameters
    dumb_ac = True,

    # Evaluation parameters
    eval = True,
    eval_env = "Simple-LTL-Env-5L-v0",
    eval_interval = 100,
    eval_samplers = ['Eventually_1_5_1_4', 'Eventually_1_5_1_4'],
    eval_episodes = [1000, 1000],
    eval_argmaxs = [True, False],
    eval_procs = 1,

    # Train parameters
    epochs = 2,
    batch_size = 1024,
    discount = 0.9,
    lr = 1e-3,
    gae_lambda = 0.5,
    entropy_coef = 0.01,
    value_loss_coef = 0.5,
    max_grad_norm = 0.5,
    clip_eps = 0.1

)