"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
#from gym.envs.mujoco import HalfCheetahEnv
#from gym.envs.mujoco import HumanoidEnv, InvertedPendulumEnv, ReacherEnv, HumanoidStandupEnv
#from gym.envs.mujoco import HopperEnv
#from gym.envs.classic_control import Continuous_MountainCarEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy, GMMPolicy, MultiTanhGaussianPolicy, SplitMultiTanhGaussianPolicy
from rlkit.torch.sac.thompsac import ThompsonSoftActorCritic
from rlkit.torch.sac.diayn import DIAYN
from rlkit.torch.networks import FlattenMlp, SplitFlattenMlp, EnsembleFlattenMlp

from custom_env import create_swingup
#from garage.envs.mujoco.maze.ant_maze_env import AntMazeEnv
#from box2d.cartpole_swingup_sparse_env import CartpoleSwingupSparseEnv

from diayn import DIAYNWrappedEnv

import torch.nn as nn

import argparse
parser     = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--heads', type=int, default=10)
parser.add_argument('--prior-size', type=int, default=128)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--prior', type=float, default=10)
parser.add_argument('--force', type=float, default=1)
parser.add_argument('--prior-offset', type=float, default=0)
parser.add_argument('--dir', type=str, default="test")
parser.add_argument('--ensemble', action='store_true')
parser.add_argument('--split-actor', action='store_true')
parser.add_argument('--split-critic', action='store_true')
args = parser.parse_args()

import torch
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

def experiment(variant):
    env = NormalizedBoxEnv(create_swingup(args.force))
    #env = NormalizedBoxEnv(HalfCheetahEnv())
    #env = NormalizedBoxEnv(Continuous_MountainCarEnv())
    #env = DIAYNWrappedEnv(NormalizedBoxEnv(HumanoidEnv()))
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

    skill_dim = 0#50
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    heads = args.heads
    
    if args.ensemble:
        print("using ensemble critic")
        
        def create_net(net_size):
            return EnsembleFlattenMlp(
                heads,
                hidden_sizes=[net_size, net_size],
                input_size=obs_dim + skill_dim + action_dim,
                output_size=1,
            )
    elif args.split_critic:
        print("using split critic")
        
        def create_net(net_size):
            return SplitFlattenMlp(
                hidden_sizes=[net_size, net_size],
                input_size=obs_dim + skill_dim + action_dim,
                output_size=1,
                heads=heads,
            )
    else:
        print("using multiheaded critic")
        
        def create_net(net_size):
            return FlattenMlp(
                hidden_sizes=[net_size, net_size],
                input_size=obs_dim + skill_dim + action_dim,
                output_size=heads,
            )
    
    qf1 = create_net(variant['net_size'])
    qf2 = create_net(variant['net_size'])
    pqf1 = create_net(args.prior_size)
    pqf2 = create_net(args.prior_size)
    
    net_size = variant['net_size']
    
    if args.split_actor:
        print("using split actor")
        policy = SplitMultiTanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim + skill_dim,
            action_dim=action_dim,
            heads=heads,
        )
    else:
        print("using multiheaded actor")
        policy = MultiTanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim + skill_dim,
            action_dim=action_dim,
            heads=heads,
        )
    algorithm = ThompsonSoftActorCritic(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        pqf1=pqf1,
        pqf2=pqf2,
        prior_coef=args.prior,
        droprate=args.drop,
        prior_offset=args.prior_offset,
        heads=heads,
        #disc=disc,
        #skill_dim=skill_dim,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__": 
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=500,
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
