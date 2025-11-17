import torch
from .storage import get_status, get_model_state
from .format import get_obss_preprocessor
from SymGroundMultiTask.ac_model import ACModel
from SymGroundMultiTask.recurrent_ac_model import RecurrentACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env, obs_space, action_space, model_dir, ignoreLTL, progression_mode, gnn,
        recurrence=1, dumb_ac=False, device=None, argmax=False, num_envs=1, verbose=True):

        if verbose:
            print(model_dir)

        status = get_status(model_dir, device)

        if status == None:
            status = {'num_frames': 0, 'update': 0}

        using_gnn = (gnn != "GRU" and gnn != "LSTM")
        obs_space, self.preprocess_obss = get_obss_preprocessor(env, using_gnn, progression_mode)
        if 'vocab' in status and self.preprocess_obss.vocab is not None:
            self.preprocess_obss.vocab.load_vocab(status['vocab'])

        if recurrence > 1:
            self.acmodel = RecurrentACModel(env, obs_space, action_space, ignoreLTL, gnn, dumb_ac, True, verbose=verbose)
            self.memories = torch.zeros(num_envs, self.acmodel.memory_size, device=device)
        else:
            self.acmodel = ACModel(env, obs_space, action_space, ignoreLTL, gnn, dumb_ac, True, device, verbose=verbose)

        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        self.acmodel.load_state_dict(get_model_state(model_dir, device))
        self.acmodel.to(self.device)
        self.acmodel.eval()


    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=False)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()


    def get_action(self, obs):
        return self.get_actions([obs])[0]


    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks


    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])