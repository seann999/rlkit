"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from gym.envs.classic_control import Continuous_MountainCarEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy, GMMPolicy, MultiTanhGaussianPolicy
from rlkit.torch.sac.bigthompsac import BigThompsonSoftActorCritic
from rlkit.torch.sac.diayn import DIAYN
from rlkit.torch.networks import EnsembleFlattenMlp, FlattenMlp

from box2d.cartpole_swingup_sparse_env import CartpoleSwingupSparseEnv

from diayn import DIAYNWrappedEnv

import argparse
parser     = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dir', type=str, default="test")
args = parser.parse_args()

import torch
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

def experiment(variant):
    env = NormalizedBoxEnv(CartpoleSwingupSparseEnv())

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    heads = 5

    net_size = variant['net_size']
    qf1 = EnsembleFlattenMlp(
        heads,
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = EnsembleFlattenMlp(
        heads,
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    pqf1 = EnsembleFlattenMlp(
        heads,
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    pqf2 = EnsembleFlattenMlp(
        heads,
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[1],
        input_size=obs_dim,
        output_size=1,
    )
    policy = MultiTanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
        heads=heads,
    )

    algorithm = BigThompsonSoftActorCritic(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        pqf1=pqf1,
        pqf2=pqf2,
        prior_coef=10,
        vf=vf,
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
            num_epochs=1000,
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
            vf_lr=3E-4,
        ),
        net_size=128,
    )
    setup_logger(args.dir, variant=variant)
    experiment(variant)
