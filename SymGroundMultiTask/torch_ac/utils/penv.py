from torch.multiprocessing import Process, Pipe
import gym

import utils


def worker(conn, env, seed):

    utils.set_seed(seed)

    while True:

        cmd, data = conn.recv()

        if cmd == "step":
            obs, reward, done, info = env.step(data)
            info = {"env_info": info, "last_obs": None}
            if done:
                info["last_obs"] = obs
                obs = env.reset()
            conn.send((obs, reward, done, info))

        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)

        elif cmd == "kill":
            return

        else:
            raise NotImplementedError



class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.closed = False

        self.locals = []
        self.processes = []

        for worker_id, env in enumerate(self.envs[1:]):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, worker_id))
            p.daemon = False
            p.start()
            self.processes.append(p)
            remote.close()


    def __del__(self):
        self.close()


    def check_closed(self):
        if self.closed:
            raise RuntimeError("Attempt to use ParallelEnv after it has been closed.")


    def close(self):
        if not self.closed:
            for local in self.locals:
                local.send(("kill", None))
            for p in self.processes:
                p.join()
            self.locals = []
            self.processes = []
            self.closed = True


    def reset(self):
        self.check_closed()
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results


    def step(self, actions):

        self.check_closed()
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))

        obs, reward, done, info = self.envs[0].step(actions[0])
        info = {"env_info": info, "last_obs": None}
        if done:
            info["last_obs"] = obs
            obs = self.envs[0].reset()

        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results


    def render(self):
        raise NotImplementedError