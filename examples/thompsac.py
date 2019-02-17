


"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.mujoco import HumanoidEnv, InvertedPendulumEnv, ReacherEnv, HumanoidStandupEnv
from gym.envs.mujoco import HopperEnv
from gym.envs.classic_control import Continuous_MountainCarEnv

<<<<<<< HEAD
=======
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy, GMMPolicy, MultiTanhGaussianPolicy
from rlkit.torch.sac.thompsac import ThompsonSoftActorCritic
from rlkit.torch.sac.diayn import DIAYN
from rlkit.torch.networks import FlattenMlp

>>>>>>> parent of 3037648... synced thompsac
#from create_maze_env import create_maze_env
from garage.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from custom_env import create_swingup

from diayn import DIAYNWrappedEnv

import argparse
parser     = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--prior', type=float, default=10)
parser.add_argument('--prior-offset', type=float, default=0)
parser.add_argument('--dir', type=str, default="test")
args = parser.parse_args()

import torch
from rlkit.launchers.launcher_util import setup_logger
#torch.manual_seed(args.seed)


def experiment(variant):
<<<<<<< HEAD
    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.wrappers import NormalizedBoxEnv
    
    from rlkit.torch.sac.policies import TanhGaussianPolicy, GMMPolicy, MultiTanhGaussianPolicy
    from rlkit.torch.sac.thompsac import ThompsonSoftActorCritic
    from rlkit.torch.sac.diayn import DIAYN
    from rlkit.torch.networks import FlattenMlp, SplitFlattenMlp

    env = NormalizedBoxEnv(CartpoleSwingupSparseEnv())
=======
    env = NormalizedBoxEnv(create_swingup())
>>>>>>> parent of 3037648... synced thompsac
    #env = NormalizedBoxEnv(HalfCheetahEnv())
    #env = NormalizedBoxEnv(Continuous_MountainCarEnv())
    #env = DIAYNWrappedEnv(NormalizedBoxEnv(HumanoidEnv()))
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

    skill_dim = 0#50
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    heads = 1

    net_size = variant['net_size']
<<<<<<< HEAD
    qf1s = [FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=1,
    ) for _ in range(heads)]
    qf2s = [FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=1,
    ) for _ in range(heads)]
    pqf1s = [FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=1,
    ) for _ in range(heads)]
    pqf2s = [FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=1,
    ) for _ in range(heads)]
    policies = [TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + skill_dim,
        action_dim=action_dim,
    ) for _ in range(heads)]
=======
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=heads,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=heads,
    )
    pqf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=heads,
    )
    pqf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=heads,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + skill_dim,
        output_size=1,
    )
    policy = MultiTanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + skill_dim,
        action_dim=action_dim,
        heads=heads,
    )
    disc = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=skill_dim if skill_dim > 0 else 1,
    )
>>>>>>> parent of 3037648... synced thompsac
    algorithm = ThompsonSoftActorCritic(
        env=env,
        policies=policies,
        qf1s=qf1s,
        qf2s=qf2s,
        pqf1s=pqf1s,
        pqf2s=pqf2s,
        prior_coef=args.prior,
        droprate=args.drop,
        prior_offset=args.prior_offset,
        heads=heads,

        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__": 
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    #mp.set_start_method('spawn')
    
    torch.backends.cudnn.deterministic = True
    
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=300,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            reward_scale=1,
            use_automatic_entropy_tuning = False,
            train_policy_with_reparameterization=True,
            min_num_steps_before_training=1000,
            replay_buffer_size=int(1e6),

            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
        ),
        net_size=128,
    )
    setup_logger(args.dir, variant=variant)
    experiment(variant)
