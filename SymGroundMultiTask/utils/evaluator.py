import time
import torch
import tensorboardX
import argparse
import datetime
from .storage import get_model_dir
from .env import make_env
from .agent import Agent
from .other import synthesize, average_discounted_return
from torch_ac.utils.penv import ParallelEnv


"""
This class evaluates a model on a validation dataset generated online
via the sampler (ltl_sampler) that is passed in (model_name).
"""
class Eval:

    def __init__(self, env, model_dir, ltl_sampler, seed=0, device="cpu", state_type='image', grounder=None,
        obs_size=None, argmax=False, num_procs=1, ignoreLTL=False, progression_mode=True, gnn=None, recurrence=1,
        dumb_ac=False, max_num_steps=None):

        self.env = env
        self.device = device
        self.argmax = argmax
        self.num_procs = num_procs
        self.ignoreLTL = ignoreLTL
        self.progression_mode = progression_mode
        self.gnn = gnn
        self.recurrence = recurrence
        self.dumb_ac = dumb_ac

        self.model_dir = model_dir
        self.eval_name = (
            "/eval-"
            + ltl_sampler
            + ("-argmax" if self.argmax else "")
        )
        self.eval_dir = self.model_dir + self.eval_name

        self.tb_writer = tensorboardX.SummaryWriter(self.eval_dir)

        # Load environments for evaluation
        eval_envs = []
        for i in range(self.num_procs):
            eval_envs.append(make_env(
                env_key = env,
                progression_mode = progression_mode,
                ltl_sampler = ltl_sampler,
                seed = seed,
                intrinsic = 0,
                noLTL = False,
                state_type = state_type,
                grounder = grounder,
                obs_size = obs_size
            ))

        if max_num_steps is not None:
            for env in eval_envs:
                env.env.max_num_steps = max_num_steps

        self.eval_envs = eval_envs
        self.eval_env = ParallelEnv(eval_envs)


    def eval(self, episodes=100):

        # Load agent
        agent = Agent(self.eval_env.envs[0], self.eval_env.observation_space, self.eval_env.action_space,
                      self.model_dir, self.ignoreLTL, self.progression_mode, self.gnn, self.recurrence, self.dumb_ac,
                      self.device, self.argmax, self.num_procs, False)

        # Run agent
        start_time = time.time()

        obss = self.eval_env.reset()
        log_counter = 0

        log_episode_return = torch.zeros(self.num_procs, device=self.device)
        log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        # Initialize logs
        logs = {"num_frames_per_episode": [], "return_per_episode": []}
        while log_counter < episodes:
            actions = agent.get_actions(obss)
            obss, rewards, dones, _ = self.eval_env.step(actions)
            agent.analyze_feedbacks(rewards, dones)

            log_episode_return += torch.tensor(rewards, device=self.device, dtype=torch.float)
            log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done in enumerate(dones):
                if done:
                    log_counter += 1
                    logs["return_per_episode"].append(log_episode_return[i].item())
                    logs["num_frames_per_episode"].append(int(log_episode_num_frames[i].item()))

            mask = 1 - torch.tensor(dones, device=self.device, dtype=torch.float)
            log_episode_return *= mask
            log_episode_num_frames *= mask

        end_time = time.time()

        return logs["return_per_episode"], logs["num_frames_per_episode"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ltl-sampler", default="Default",
                    help="the ltl formula template to sample from (default: DefaultSampler)")
    parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
    parser.add_argument("--model-paths", required=True, nargs="+",
                    help="path of the model, or a regular expression")
    parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 1)")
    parser.add_argument("--eval-episodes", type=int,  default=5,
                    help="number of episodes to evaluate on (default: 5)")
    parser.add_argument("--env", default="Letter-7x7-v3",
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--ignoreLTL", action="store_true", default=False,
                    help="the network ignores the LTL input")
    parser.add_argument("--progression-mode", default="full",
                    help="Full: uses LTL progression; partial: shows the propositions which progress or falsify the formula; none: only original formula is seen. ")
    parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--gnn", default="RGCN_8x32_ROOT_SHARED", help="use gnn to model the LTL (only if ignoreLTL==True)")

    args = parser.parse_args()

    logs_returns_per_episode = []
    logs_num_frames_per_episode = [] 

    for model_path in args.model_paths:
        idx = model_path.find("seed:") + 5
        seed = int(model_path[idx:idx+2].strip("_"))

        eval = Eval(args.env, model_path, args.ltl_sampler,
                     seed=seed, device=torch.device("cpu"), argmax=False,
                     num_procs=args.procs, ignoreLTL=args.ignoreLTL, progression_mode=args.progression_mode, gnn=args.gnn, recurrence=args.recurrence, dumb_ac=False, discount=args.discount)
        rpe, nfpe = eval.eval(-1, episodes=args.eval_episodes)
        logs_returns_per_episode += rpe
        logs_num_frames_per_episode += nfpe 

        print(sum(rpe), seed, model_path)

    print(logs_num_frames_per_episode)
    print(logs_returns_per_episode)
    num_frame_pe = sum(logs_num_frames_per_episode)
    return_per_episode = synthesize(logs_returns_per_episode)
    num_frames_per_episode = synthesize(logs_num_frames_per_episode)
    average_discounted_return, error = average_discounted_return(logs_returns_per_episode, logs_num_frames_per_episode, args.discount, include_error=True)

    header = ["frames"]
    data   = [num_frame_pe]
    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
    data += num_frames_per_episode.values()
    header += ["average_discounted_return", "err"]
    data += [average_discounted_return, error]

    header += ["return_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()

    for field, value in zip(header, data):
        print(field, value)