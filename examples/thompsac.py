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

from torch.nn import functional as F
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy, GMMPolicy, MultiTanhGaussianPolicy, SplitMultiTanhGaussianPolicy
from rlkit.torch.sac.thompsac import ThompsonSoftActorCritic
from rlkit.torch.sac.diayn import DIAYN
from rlkit.torch.networks import FlattenMlp, SplitFlattenMlp, EnsembleFlattenMlp

from custom_env import create_swingup
import pickle
#from garage.envs.mujoco.maze.ant_maze_env import AntMazeEnv
#from box2d.cartpole_swingup_sparse_env import CartpoleSwingupSparseEnv

from diayn import DIAYNWrappedEnv

import torch.nn as nn

import argparse
parser     = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--heads', type=int, default=10)
parser.add_argument('--net-size', type=int, default=256)
parser.add_argument('--prior-size', type=int, default=256)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--prior', type=float, default=10)
parser.add_argument('--force', type=float, default=1)
parser.add_argument('--reward-scale', type=float, default=1)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--int-w', type=float, default=0.1)
parser.add_argument('--int-discount', type=float, default=0.99)
parser.add_argument('--int-direct', action='store_true')
parser.add_argument('--dir', type=str, default="test")
parser.add_argument('--env', type=str, default="line")
parser.add_argument('--ensemble', action='store_true')
parser.add_argument('--split-actor', action='store_true')
parser.add_argument('--split-critic', action='store_true')
parser.add_argument('--range-prior', action='store_true')
parser.add_argument('--autotune', action='store_true')
parser.add_argument('--new', action='store_true')
parser.add_argument('--rnd', action='store_true')

parser.add_argument('--norm-obs', action='store_true')
parser.add_argument('--norm-int-r', action='store_true')

parser.add_argument('--load-prior', type=str, default=None)
parser.add_argument('--load-policy', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--activation', type=str, default="elu")
args = parser.parse_args()

from line import LineEnv

import torch
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.core import PyTorchModule

class CombineMlp(PyTorchModule):
    def __init__(self, mlp1, mlp2):
        self.save_init_params(locals())
        super().__init__()
        
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def forward(self, *inputs, **kwargs):
        return self.mlp1(*inputs) + 10 * self.mlp2(*inputs)

def experiment(variant):
    if args.env == "line":
        env = NormalizedBoxEnv(LineEnv())
    elif args.env == "swingup":
        env = NormalizedBoxEnv(create_swingup(args.force))
    else:
        import gym
        env = NormalizedBoxEnv(gym.make(args.env))

    #env = NormalizedBoxEnv(HalfCheetahEnv())
    #env = NormalizedBoxEnv(Continuous_MountainCarEnv())
    #env = DIAYNWrappedEnv(NormalizedBoxEnv(HumanoidEnv()))
    # Or for a specific version:
    

    skill_dim = 0#50
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    heads = args.heads
    
    if args.activation == "selu":
        hidden_act = F.selu
    elif args.activation == "relu":
        hidden_act = F.relu
    elif args.activation == "elu":
        hidden_act = F.elu
    elif args.activation == "tanh":
        hidden_act = F.tanh
    
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
                hidden_activation=hidden_act,
            )
    
    qf1 = create_net(variant['net_size'])
    qf2 = create_net(variant['net_size'])
    qfB = FlattenMlp(
        hidden_sizes=[variant['net_size'], variant['net_size']],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=2,
        hidden_activation=hidden_act,
    )
    qfC = FlattenMlp(
        hidden_sizes=[variant['net_size'], variant['net_size']],
        input_size=obs_dim + skill_dim + action_dim,
        output_size=1,
        hidden_activation=hidden_act,
    )
    pqf1 = create_net(args.prior_size)
    pqf2 = create_net(args.prior_size)
    
    if args.load_prior:
        print("loading prior", args.load_prior)
        data = pickle.load(open(args.load_prior, "rb"))
        pqf1 = CombineMlp(data['qf1'], data['pqf1'])
        pqf2 = CombineMlp(data['qf2'], data['pqf2'])
    
    net_size = variant['net_size']
    
    if args.split_actor:
        print("using split actor")
        policy = SplitMultiTanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim + skill_dim,
            action_dim=action_dim,
            heads=heads,
            hidden_activation=hidden_act,
        )
    else:
        print("using multiheaded actor")
        policy = MultiTanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim + skill_dim,
            action_dim=action_dim,
            heads=heads,
            hidden_activation=hidden_act,
        )
        policyB = MultiTanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim + skill_dim,
            action_dim=action_dim,
            heads=2,
            hidden_activation=hidden_act,
        )
        policyC = MultiTanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim + skill_dim,
            action_dim=action_dim,
            heads=2,
            hidden_activation=hidden_act,
        )
        
    if args.load_policy:
        print("loading policy")
        policy = data['policy']
    
    if args.range_prior:
        coefs = [0, 0.01, 0.03, 0.1, 0.3, 1, 2, 4, 8, 16, 32, 64]
        prior = np.array(coefs[:heads], dtype=np.float32)
        prior = torch.FloatTensor(prior).cuda()
    else:
        prior = args.prior
    
    algorithm = ThompsonSoftActorCritic(
        env=env,
        policy=policy,
        policyB=policyB,
        policyC=policyC,
        qf1=qf1,
        qf2=qf2,
        qfB=qfB,
        qfC=qfC,
        pqf1=pqf1,
        pqf2=pqf2,
        prior_coef=prior,
        droprate=args.drop,
        heads=heads,
        #disc=disc,
        #skill_dim=skill_dim,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__": 
    
    if args.env == "line":
        maxpath = 110
        evalsteps = 1000
        epochsteps = 200
        numepochs = 500000
    elif args.env == "swingup":
        maxpath = 1000
        evalsteps = 1000
        epochsteps = 1000
        numepochs = 500
    else:
        maxpath = 1000
        evalsteps = 1000
        epochsteps = 1000
        numepochs = 1000
        
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=numepochs,
            num_steps_per_epoch=epochsteps,
            num_steps_per_eval=evalsteps,
            batch_size=128,
            max_path_length=maxpath,
            discount=0.99,
            int_w=args.int_w,
            int_discount=args.int_discount,
            int_direct=args.int_direct,
            newmethod=args.new,
            reward_scale=args.reward_scale,
            alpha=args.alpha,
            rnd=args.rnd,
            norm_int_r=args.norm_int_r,
            norm_obs=args.norm_obs,
            use_automatic_entropy_tuning = args.autotune,
            train_policy_with_reparameterization=True,
            min_num_steps_before_training=1000,
            replay_buffer_size=int(1e6),
            
            policy_mean_reg_weight=0,
            #policy_std_reg_weight=0,
            #policy_pre_activation_weight=0,

            soft_target_tau=args.tau,
            policy_lr=args.lr,
            qf_lr=args.lr,
        ),
        net_size=args.net_size,
    )
    setup_logger(args.dir, variant=variant)
    experiment(variant)
