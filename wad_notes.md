# Atari games

## 13/01/2025

- setup a legacy environment

    ```bash
        conda create -n py3.7 python=3.7
        pip install tensorflow-gpu==1.15.0
        uv pip install -r req.txt
        %  Allow gym moving dot
        
    ```
## Run command options

### Command Line Arguments

- `--mode` (str): Specifies the mode of operation. Choices are:
    - `gather_initial_prefs`
    - `pretrain_reward_predictor`
    - `train_policy_with_preferences`
    - `train_policy_with_original_rewards`
- `--env` (str): Specifies the environment to use. Choices are
    - 'MovingDot-v0various techniques for learning from human feedback to improve machine learning models.', 
    - 'MovingDotNoFrameskip-v0',
    - 'PongNoFrameskip-v4',
    - 'EnduroNoFrameskip-v4'

Optional arguments:
- `--test_mode` (bvarious techniques for learning from human feedback to improve machine learning models.ool): Enable test mode.
- `--debug` (bool): Enable debug mode.
- `--render_episodes` (bool): Render episodes during execution.
- `--load_prefs_dir` (str): Directory to load preferences from.
- `--n_initial_prefs` (int): Number of initial preferences to collect from a random policy before starting reward predictor training. Default is 500.
- `--max_prefs` (int): Maximum number of preferences to maintain in the buffer. Default is 3000.

Logging options (mutually exclusive): 
- `--log_dir` (str): Directory to save logs.
- `--run_name` (str): Name of the run. Default is the current timestamp.

### A2C Arguments

- `--log_interval` (int): Interval between logging. Default is 100.
- `--ent_coef` (float): Entropy coefficient. Default is 0.01.
- `--n_envs` (int): Number of environments. Default is 1.
- `--seed` (int): RNG seed. Default is 0.
- `--lr_zero_million_timesteps` (float): If set, decay learning rate linearly, reaching zero at this many timesteps. Default is None.
- `--lr` (float): Learning rate. Default is 7e-4.
- `--load_policy_cvarious techniques for learning from human feedback to improve machine learning models.
### Reward Predictor Arguments

- `--reward_predictor_learning_rate` (float): Learning rate for the reward predictor. Default is 2e-4.
- `--n_initial_epochs` (int): Number of initial epochs. Default is 200.
- `--dropout` (float): Dropout rate. Default is 0.0.
- `--batchnorm` (bool): Enable batch normalization.
- `--load_reward_predictor_ckpt_dir` (str): Directory to load reward predictor checkpoint from (loads latest checkpoint in the specified directory).
- `--reward_predictor_ckpt_interval` (int): Number of training epochs between reward predictor checkpoints. Default is 1.

### Preference interface Arguments

- `--synthetic_perfs` (?)
- `--max_segs` (int) Maximum number of segments to store. Default is 1000.




# Our actions

## Reproducing
1. Train pong with original rewards: runs/1736865979_3fca07c
`$ python3 run.py train_policy_with_original_rewards PongNoFrameskip-v4 --n_envs 16 --million_timesteps 10`
Saved policy checkpoint to 'runs/1736865979_3fca07c/policy_checkpoints/policy.ckpt-125000'

2. Visualize the results with tensorboard in a notebook:
`%%tensorboard`
`%tensorboard --logdir runs/1736865979_3fca07c`



3. Train with human preferences: 
Enter 'L' in the terminal to indicate that you prefer the left example; 'R' to indicate
you prefer the right example; 'E' to indicate you prefer them both equally; and
just press enter if the two clips are incomparable.


```SHELL
Trained policy for 24000 time steps
Saved policy checkpoint to 'runs/1737100777_3fca07c/policy_checkpoints/policy.ckpt-300'
Saved training preferences to 'runs/1737100777_3fca07c/train.pkl.gz'
Saved validation preferences to 'runs/1737100777_3fca07c/val.pkl.gz'
Training/testing with 650/155 preferences
Saved training preferences to 'runs/1737100777_3fca07c/train.pkl.gz'
Saved validation preferences to 'runs/1737100777_3fca07c/val.pkl.gz'
Training/testing with 650/155 preferences
```

**Run policy**
python3 run_checkpoint.py PongNoFrameskip-v4 runs/1737100777_3fca07c/policy_checkpoints/

 

 ### Train the moving dot
 ```shell
 python3 run.py train_policy_with_preferences MovingDotNoFrameskip-v0 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15
 ```


 ### Train Pong

 #### With human preferences
 Gather the initial 500 preferences for Pong
 ```bash
 python run.py gather_initial_prefs PongNoFrameskip-v4 --dropout 0.5 --n_envs 16 --million_timesteps 20 --render_episodes --run_name Pong_init_perf
>>> Saved training preferences to 'runs/Pong_init_perf_4408c85/train.pkl.gz'
>>> Saved validation preferences to 'runs/Pong_init_perf_4408c85/val.pkl.gz'
``` 



```bash
python3 run.py pretrain_reward_predictor PongNoFrameskip-v4 --load_prefs_dir runs/Pong_init_perf_4408c85
```

#### Base line
For comparison, train with synthetic (oracle) preference.
```bash
python3 run.py train_policy_with_preferences PongNoFrameskip-v4 --synthetic_prefs --dropout 0.5 --n_envs 16 --million_timesteps 20
```
Saved reward predictor checkpoint to 'runs/1738594711_4408c85/reward_predictor_checkpoints/reward_predictor.ckpt-3200'
Saved training preferences to 'runs/1738595245_4408c85/train.pkl.gz'
Saved validation preferences to 'runs/1738595245_4408c85/val.pkl.gz'

```bash
python3 run.py train_policy_with_original_rewards PongNoFrameskip-v4 --n_envs 16 --million_timesteps 10 --run_name pong_original

>>>Trained policy for 10000000 time steps
>>>Saved policy checkpoint to 'runs/pong_original_4408c85/policy_checkpoints/policy.ckpt-125000'
>>>Saved policy checkpoint to 'runs/pong_original_4408c85/policy_checkpoints/policy.ckpt-125000'
```


Next step : try to finetuene the previous policy
python3 run.py train_policy_with_preferences PongNoFrameskip-v4 --n_envs 16 --million_timesteps 10 --load_policy_ckpt_dir runs/pong_original_4408c85/policy_checkpoints --load_prefs_dir runs/Pong_init_perf_4408c85 --run_name pong_finetune