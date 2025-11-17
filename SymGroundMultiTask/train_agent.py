import time
import torch
import tensorboardX
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from tqdm import tqdm

import utils
import torch_ac
from ac_model import ACModel
from recurrent_ac_model import RecurrentACModel
from grounder_algo import GrounderAlgo


@dataclass
class Args:

    # General parameters
    model_name: Optional[str] = None
    algo: str = "ppo"
    seed: int = 1
    log_interval: int = 10
    save_interval: int = 100
    procs: int = 16
    frames_per_proc: Optional[int] = None
    frames: int = 2 * 10**8
    checkpoint_dir: Optional[str] = None

    # Environment parameters
    env: str = "GridWorld-fixed-v1"
    state_type: str = "image"
    obs_size: Tuple[int,int] = (56,56)
    ltl_sampler: str = "Dataset_e54"
    noLTL: bool = False
    progression_mode: str = "full"
    int_reward: float = 0.0

    # GNN parameters
    ignoreLTL: bool = False
    gnn_model: str = "RGCN_8x32_ROOT_SHARED"
    use_pretrained_gnn: bool = False
    gnn_pretrain: Optional[str] = None
    freeze_gnn: bool = False

    # Grounder parameters
    grounder_model: Optional[str] = "ObjectCNN"
    use_pretrained_grounder: bool = False
    grounder_pretrain: Optional[str] = None
    freeze_grounder: bool = False

    # Agent parameters
    dumb_ac: bool = False
    recurrence: int = 1

    # Evaluation parameters
    eval: bool = False
    eval_env: Optional[str] = None
    eval_interval: int = 100
    eval_samplers: Optional[List[str]] = None
    eval_episodes: Optional[List[int]] = None
    eval_argmaxs: Optional[List[bool]] = None
    eval_procs: int = 1

    # Train parameters
    epochs: int = 4
    batch_size: int = 256
    discount: float = 0.99
    lr: float = 0.0003
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5  # gradient clipping
    optim_eps: float = 1e-8
    optim_alpha: float = 0.99
    clip_eps: float = 0.2  # ppo clipping epsilon

    # Grounder training parameters
    grounder_buffer_size: int = 1000
    grounder_buffer_start: int = 32
    grounder_max_env_steps: int = 75
    grounder_train_interval: int = 1
    grounder_lr: float = 0.001
    grounder_batch_size: int = 32
    grounder_update_steps: int = 4
    grounder_accumulation: int = 1
    grounder_evaluate_steps: int = 1
    grounder_use_early_stopping: float = False
    grounder_patience: int = 20
    grounder_min_delta: float = 0.0


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def train_agent(args: Args, device: str = None):

    # SETUP

    use_grounder = args.grounder_model is not None
    train_grounder = use_grounder and not args.freeze_grounder
    use_mem = args.recurrence > 1
    use_gnn = (args.gnn_model != "GRU" and args.gnn_model != "LSTM")

    # check if arguments are consistent
    if args.freeze_gnn:
        assert args.use_pretrained_gnn
    if args.use_pretrained_gnn:
        assert args.progression_mode in ["full", "real"]
        assert args.gnn_pretrain is not None
    if use_grounder and args.freeze_grounder:
        assert args.use_pretrained_grounder
    if args.use_pretrained_grounder:
        assert args.grounder_pretrain is not None
    if args.eval and args.eval_samplers:
        assert len(args.eval_episodes) == len(args.eval_samplers)
    if args.eval and args.eval_argmaxs:
        assert len(args.eval_episodes) == len(args.eval_argmaxs)
    if train_grounder:
        assert args.grounder_buffer_size >= args.grounder_buffer_start
    if train_grounder and args.grounder_use_early_stopping:
        assert args.grounder_patience > 0

    device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint dirs
    storage_dir = "storage" if args.checkpoint_dir is None else args.checkpoint_dir
    storage_dir = os.path.join(REPO_DIR, storage_dir)
    pretrain_dir = os.path.join(REPO_DIR, "storage-pretrain")

    # build GNN name
    gnn_name = (
        ("IgnoreLTL" if args.ignoreLTL else args.gnn_model)
        + ("-dumb_ac" if args.dumb_ac else "")
        + ("-pretrained" if args.use_pretrained_gnn else "")
        + ("-freeze_gnn" if args.freeze_gnn else "")
        + (f"-recurrence:{args.recurrence}" if use_mem else "")
    )

    # compute default_model_name
    default_model_name = (
        f"{gnn_name}_{args.ltl_sampler}_{args.env}"
        f"_seed:{args.seed}_epochs:{args.epochs}"
        f"_bs:{args.batch_size}_fpp:{args.frames_per_proc}"
        f"_dsc:{args.discount}_lr:{args.lr}"
        f"_ent:{args.entropy_coef}_clip:{args.clip_eps}"
        f"_prog:{args.progression_mode}"
    )

    # model dir
    model_name = args.model_name or default_model_name
    model_dir = utils.get_model_dir(model_name, storage_dir)
    train_dir = os.path.join(model_dir, "train")

    # pretrained gnn dir
    pretrained_gnn_dir = None
    if args.use_pretrained_gnn:
        pretrained_gnn_dir = utils.get_model_dir(args.gnn_pretrain, pretrain_dir)

    # pretrained grounder dir
    pretrained_grounder_dir = None
    if use_grounder and args.use_pretrained_grounder:
        pretrained_grounder_dir = utils.get_model_dir(args.grounder_pretrain, pretrain_dir)

    # load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(train_dir)
    tb_writer = tensorboardX.SummaryWriter(train_dir)
    utils.save_config(model_dir, args)

    # log script arguments
    txt_logger.info("\n---\n")
    txt_logger.info("Args:")
    for field_name, value in vars(args).items():
        txt_logger.info(f"\t{field_name}: {value}")
    txt_logger.info(f"\nDevice: {device}")
    txt_logger.info("\n---\n")

    # set seed for all randomness sources
    utils.set_seed(args.seed)


    # INITIALIZATION

    txt_logger.info("Initialization\n")

    # load grounder algo environment
    grounder_algo_env = utils.make_env(args.env, args.progression_mode, args.ltl_sampler, args.seed,
                                       args.int_reward, args.noLTL, args.state_type, None, args.obs_size)

    num_symbols = len(grounder_algo_env.propositions) + 1

    # create grounder
    sym_grounder = utils.make_grounder(args.grounder_model, num_symbols, args.obs_size, args.freeze_grounder)
    grounder_algo_env.env.sym_grounder = sym_grounder

    # load environments
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.progression_mode, args.ltl_sampler, args.seed, args.int_reward,
                                   args.noLTL, args.state_type, sym_grounder, args.obs_size))

    txt_logger.info("-) Environments loaded.")

    # load previous training status
    status = utils.get_status(model_dir, device)
    txt_logger.info("-) Looking for status of previous training.")
    if status == None:
        status = {'num_frames': 0, 'update': 0}
        txt_logger.info("-) Previous status not found.")
    else:
        txt_logger.info("-) Previous status found.")

    # load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0], use_gnn, args.progression_mode)
    if 'vocab' in status and preprocess_obss.vocab is not None:
        preprocess_obss.vocab.load_vocab(status['vocab'])
    txt_logger.info("-) Observations preprocessor loaded.")

    # create model
    if use_mem:
        acmodel = RecurrentACModel(envs[0].env, obs_space, envs[0].action_space, args.ignoreLTL,
                                   args.gnn_model, args.dumb_ac, args.freeze_gnn, device, False)
    else:
        acmodel = ACModel(envs[0].env, obs_space, envs[0].action_space, args.ignoreLTL,
                          args.gnn_model, args.dumb_ac, args.freeze_gnn, device, False)

    # load existing model
    if 'model_state' in status:
        acmodel.load_state_dict(status['model_state'])
        txt_logger.info("-) Loading model from existing run.")

    # otherwise load existing pretrained GNN
    elif args.use_pretrained_gnn:
        gnn_status = utils.get_status(pretrained_gnn_dir, device)
        acmodel.load_pretrained_gnn(gnn_status['model_state'])
        txt_logger.info("-) Loading GNN from pretrain.")

    # load existing grounder
    if use_grounder and 'grounder_state' in status:
        sym_grounder.load_state_dict(status['grounder_state'])
        txt_logger.info("-) Loading grounder from existing run.")

    # otherwise load existing pretrained grounder
    elif use_grounder and args.use_pretrained_grounder:
        grounder_status = utils.get_status(pretrained_grounder_dir, device)
        sym_grounder.load_state_dict(grounder_status['grounder_state'])
        status['num_frames'] += grounder_status['num_frames']
        status['grounder_state'] = grounder_status['grounder_state']
        txt_logger.info("-) Loading grounder from pretrain.")

    sym_grounder.to(device) if sym_grounder is not None else None
    acmodel.to(device)

    txt_logger.info("-) Model loaded.")

    # load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss, None)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, None)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    # load optimizer of existing model
    if 'optimizer_state' in status:
        algo.optimizer.load_state_dict(status['optimizer_state'])
        txt_logger.info("-) Loading optimizer from existing run.")

    txt_logger.info("-) Agent training algorithm loaded.")

    # load grounder algo
    grounder_algo = GrounderAlgo(sym_grounder, grounder_algo_env, train_grounder, args.grounder_max_env_steps,
                                 args.grounder_buffer_size, args.grounder_lr, args.grounder_batch_size,
                                 args.grounder_update_steps, args.grounder_accumulation, args.grounder_evaluate_steps,
                                 args.grounder_use_early_stopping, args.grounder_patience, args.grounder_min_delta,
                                 model_dir, device)

    # load grounder optimizer of existing model
    if train_grounder and 'grounder_optimizer_state' in status:
        grounder_algo.optimizer.load_state_dict(status['grounder_optimizer_state'])
        grounder_algo.early_stop = status['grounder_early_stop']
        txt_logger.info("-) Loading grounder optimizer from existing run.")

    elif train_grounder and args.use_pretrained_grounder:
        grounder_algo.optimizer.load_state_dict(grounder_status['grounder_optimizer_state'])
        grounder_algo.early_stop = grounder_status['grounder_early_stop']
        status['grounder_optimizer_state'] = grounder_status['grounder_optimizer_state']
        status['grounder_early_stop'] = grounder_status['grounder_early_stop']
        txt_logger.info("-) Loading grounder optimizer from pretrain.")

    txt_logger.info("-) Grounder training algorithm loaded.")

    # initialize the evaluators
    evals = []
    if args.eval:

        eval_env = args.eval_env if args.eval_env else args.env
        eval_samplers = args.eval_samplers if args.eval_samplers else [args.ltl_sampler]
        eval_argmaxs = args.eval_argmaxs if args.eval_argmaxs else [True for _ in range(len(eval_samplers))]
        eval_procs = args.eval_procs if args.eval_procs else 1

        for sampler, argmax in zip(eval_samplers, eval_argmaxs):
            evals.append(utils.Eval(eval_env, model_dir, sampler, args.seed, device, args.state_type, sym_grounder,
                                    args.obs_size, argmax, eval_procs, args.ignoreLTL, args.progression_mode,
                                    args.gnn_model, args.recurrence, args.dumb_ac, None))

        txt_logger.info("-) Evaluators loaded.")

    txt_logger.info("\n---\n")


    # TRAINING

    txt_logger.info("Training\n")

    logs1 = utils.empty_episode_logs()
    logs2 = utils.empty_buffer_logs()
    logs3 = utils.empty_algo_logs()
    logs4 = utils.empty_grounder_algo_logs()
    logs5 = utils.empty_grounder_eval_logs(num_symbols)
    logs_exp = utils.empty_episode_logs()

    num_frames = status['num_frames']
    update = status['update']
    start_time = time.time()

    # populate buffer
    if train_grounder and not status['grounder_early_stop']:
        txt_logger.info("Initializing Buffer...\n")
        progress = tqdm(total=args.grounder_buffer_start)
        while progress.n < args.grounder_buffer_start:
            logs = grounder_algo.collect_experiences()
            progress.n = logs['buffer']
            num_frames += logs['episode_frames']
            progress.refresh()
        progress.close()

    # training loop
    while num_frames < args.frames:

        update_start_time = time.time()
        update += 1

        # collect experiences by playing in the environments
        exps, logs1 = algo.collect_experiences()
        logs2 = grounder_algo.process_experiences(exps)
        num_frames += logs1['num_frames']
        logs_exp = utils.accumulate_episode_logs(logs_exp, logs1)

        # updated agent
        logs3 = algo.update_parameters(exps)

        # update grounder
        if update % args.grounder_train_interval == 0:
            logs4 = grounder_algo.update_parameters()

        update_end_time = time.time()

        # Print logs (accumulated during the log_interval)

        if (update % args.log_interval == 0) or (num_frames >= args.frames):

            fps = logs1['num_frames']/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)

            logs1 = utils.elaborate_episode_logs(logs_exp, args.discount)
            logs5 = grounder_algo.evaluate()
            logs = {**logs1, **logs2, **logs3, **logs4, **logs5}
            logs_exp = utils.empty_episode_logs()

            header = ['time/update', 'time/frames', 'time/fps', 'time/duration']
            data = [update, num_frames, fps, duration]
            header += ['return/' + key for key in logs['return_per_episode'].keys()]
            data += logs['return_per_episode'].values()
            header += ['average_discounted_return']
            data += [logs['average_discounted_return']]
            header += ['episode_frames/' + key for key in logs['num_frames_per_episode'].keys()]
            data += logs['num_frames_per_episode'].values()
            header += ['algo/entropy', 'algo/value', 'algo/policy_loss', 'algo/value_loss', 'algo/grad_norm']
            data += [logs['entropy'], logs['value'], logs['policy_loss'], logs['value_loss'], logs['grad_norm']]
            header += ['grounder/loss', 'grounder/val_loss', 'grounder/acc', 'grounder/buffer']
            data += [logs['grounder_loss'], logs['grounder_val_loss'], logs['grounder_acc'], logs['buffer']]

            # μ: mean | σ: std | m: min | M: max
            # U: update | tF: total frames | FPS | D: duration | R: return | ADR: average discounted return
            # F: episode frames | H: entropy | V: value | pL: policy loss | vL: value loss 
            # nabla: grad norm | gL: grounder loss | gvL: grounder validation loss | gA: grounder accuracy | b: buffer
            txt_logger.info(
                ("U {:5} | tF {:7.0f} | FPS {:4.0f} | D {:5} | R:μσmM {:5.2f} {:5.2f} {:5.2f} {:5.2f} | ADR {:6.3f}" +
                " | eF:μσmM {:4.1f} {:4.1f} {:2.0f} {:2.0f} | H {:.3f} | V {:6.3f} | pL {:6.3f} | vL {:.3f}" +
                " | ∇ {:.3f} | gL {:.6f} | gvL {:.6f} | gA {:.4f} | b {:5}").format(*data)
            )

            header += ['average_reward_per_step', 'average_discounted_return']
            data += [logs['average_reward_per_step'], logs['average_discounted_return']]

            header += ['grounder/buffer_val', 'grounder/total_buffer', 'grounder/total_buffer_val']
            data += [logs['val_buffer'], logs['total_buffer'], logs['total_val_buffer']]

            header += [f'grounder_recall/{i}' for i in range(num_symbols)]
            data += logs['grounder_recall']

            if status['num_frames'] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        eval_condition = ((args.eval and args.eval_interval > 0 and update % args.eval_interval == 0)
                          or (args.eval and num_frames >= args.frames)
                          or (args.eval and update == 1))

        save_condition = ((args.save_interval > 0 and update % args.save_interval == 0)
                          or (eval_condition)
                          or (num_frames >= args.frames))

        # Save status

        if save_condition:

            status['num_frames'] = num_frames
            status['update'] = update
            status['model_state'] = algo.acmodel.state_dict()
            status['optimizer_state'] = algo.optimizer.state_dict()

            if train_grounder:
                status['grounder_state'] = sym_grounder.state_dict()
                status['grounder_optimizer_state'] = grounder_algo.optimizer.state_dict()
                status['grounder_early_stop'] = grounder_algo.early_stop

            if hasattr(preprocess_obss, 'vocab') and preprocess_obss.vocab is not None:
                status['vocab'] = preprocess_obss.vocab.vocab

            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        # Compute Evaluation

        if eval_condition:

            for i, evalu in enumerate(evals):

                eval_start_time = time.time()
                return_per_episode, frames_per_episode = evalu.eval(args.eval_episodes[i])
                eval_end_time = time.time()

                duration = int(eval_end_time - eval_start_time)

                total_eval_frames = sum(frames_per_episode)
                average_discounted_return = utils.average_discounted_return(return_per_episode, frames_per_episode, args.discount)
                return_per_episode = utils.synthesize(return_per_episode)
                frames_per_episode = utils.synthesize(frames_per_episode)

                header = ['time/frames', 'time/duration']
                data = [total_eval_frames, duration]
                header += ['return/' + key for key in return_per_episode.keys()]
                data += return_per_episode.values()
                header += ['average_discounted_return']
                data += [average_discounted_return]
                header += ['episode_frames/' + key for key in frames_per_episode.keys()]
                data += frames_per_episode.values()

                txt_logger.info(f"Evaluator {i} ({evalu.eval_name})")
                txt_logger.info(
                    ("F {:7.0f} | D {:5} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | ADR {:.3f}" +
                    " | F:μσmM {:4.1f} {:4.1f} {:2.0f} {:2.0f}").format(*data)
                )

                for field, value in zip(header, data):
                    evalu.tb_writer.add_scalar(field, value, num_frames)


    # TERMINATION

    # close loggers
    tb_writer.close()
    for evalu in evals:
        evalu.tb_writer.close()
    utils.close_txt_logger(txt_logger)
    csv_file.close()

    # kill subprocesses
    algo.env.close()
    for evalu in evals:
        evalu.eval_env.close()