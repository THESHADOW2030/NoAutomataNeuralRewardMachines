# Neural Reward Machines
Repository under development for the paper "Fully Learnable Neural Reward Machines. Hazem Dewidar and Elena Umili. 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis (OVERLAY@ECAI2025) ".

## Requirements
- pytorch
- gym
- pygame

## How reproduce the experiments
To reproduce the experiments in the paper run the script ```experiments.py```

The script uses some flags, run ``` python experiments.py --helpfull``` to see the full list of parameters used.

```
       USAGE: experiments.py [flags]
flags:

experiments.py:
  --ENV: Environment to test, one in ['map_env', 'image_env'], default= 'map_env'
    (default: 'map_env')
  --LOG_DIR: path where to save the results, default='Results/'
    (default: 'Results/')
  --METHOD: Method to test, one in ['rnn', 'nrm', 'rm'], default= 'rnn'
    (default: 'rnn')
  --NUM_EXPERIMENTS: num of runs for each test, default= 5
    (default: '5')
    (an integer)
```
## Citations
```
@misc{dewidar2025fullylearnableneuralreward,
      title={Fully Learnable Neural Reward Machines}, 
      author={Hazem Dewidar and Elena Umili},
      year={2025},
      eprint={2509.19017},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.19017}, 
}
```

