from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn

import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic

import torch.nn.functional as F

from torch.nn.parallel import parallel_apply

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from gym.spaces import Box, Discrete, Tuple

class CustomReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            obs_dims,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        #self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=obs_dims,
            action_dim=get_dim(self._action_space),
        )

    def add_sample(self, observation, action, reward, terminal,
            next_observation, **kwargs):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        super(CustomReplayBuffer, self).add_sample(
                observation, action, reward, terminal, 
                next_observation, **kwargs)


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


class RNDSoftActorCritic(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            rf,
            pf,
            vf,

            int_coef=1,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            train_policy_with_reparameterization=True,
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            replay_buffer_size=1000000,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            **kwargs
    ):
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
            
        self.heads = 10
        replay_buffer = CustomReplayBuffer(
            replay_buffer_size,
            env,
            get_dim(env.observation_space) + self.heads
        )
            
        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            replay_buffer=replay_buffer,
            **kwargs
        )

        self.int_coef
        self.policy = policy
        self.pf_std = None
        self.obs_mean = None
        self.obs_std = None
        self.qf1 = qf1
        self.qf2 = qf2
        self.rf = rf
        self.pf = pf
        self.vf = vf
        self.minibuffer = []
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.target_vf = vf.copy()
        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            qf2.parameters(),
            lr=qf_lr,
        )
        self.pf_optimizer = optimizer_class(
            pf.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        
    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        if self.heads == 1:
            mask = [1]
        else:
            mask = np.random.randint(2, size=self.heads)
            #mask = [1] * self.heads
            
        self.minibuffer.append(observation)
        if len(self.minibuffer) >= 128:
            obsbatch = torch.FloatTensor(np.array(self.minibuffer)).cuda()
            
            if self.obs_mean is None:
                self.obs_mean = obsbatch.mean(0)
                self.obs_std = obsbatch.std(0)
                self.obs_std[self.obs_std == 0] = 1
                
            if self.pf_std is None:
                p_pred = self.pf((obsbatch - self.obs_mean) / self.obs_std)
                r_feats = self.rf((obsbatch - self.obs_mean) / self.obs_std)
                pf_loss = ((p_pred - r_feats.detach())**2.0).sum(1)
            
                #if self.pf_std is None:
                self.pf_std = pf_loss.std()
            
            #self.pf_optimizer.zero_grad()
            #pf_loss.mean().backward()
            #self.pf_optimizer.step()
            
            self.minibuffer = []
            
        observation = np.concatenate([observation, mask])
        next_observation = np.concatenate([next_observation, np.zeros(self.heads)])
        
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
        )


    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations'][:, :-self.heads]
        actions = batch['actions']
        next_obs = batch['next_observations'][:, :-self.heads]
        
        self.obs_mean = 0.99 * self.obs_mean + 0.01 * obs.mean(0)
        std = obs.std(0)
        std[std == 0] = 1
        self.obs_std = 0.99 * self.obs_std + 0.01 * std

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        r_feats = self.rf((obs - self.obs_mean) / self.obs_std)
        mask = batch['observations'][:, -self.heads:]
        
        #v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(
                obs,
                reparameterize=self.train_policy_with_reparameterization,
                return_log_prob=True,
        )
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """
        QF Loss
        """
        
        #target_v_values = self.target_vf(next_obs)
        next_policy_outputs = self.policy(
                next_obs,
                reparameterize=self.train_policy_with_reparameterization,
                return_log_prob=True,
        )
        next_new_actions, _, _, next_log_pi = next_policy_outputs[:4]
        target_qf1 = self.target_qf1(next_obs, next_new_actions)
        target_qf2 = self.target_qf2(next_obs, next_new_actions)

        next_q_new_actions = torch.min(target_qf1, target_qf2)
        target_v_values = next_q_new_actions - alpha*next_log_pi
        
        p_pred = self.pf((obs - self.obs_mean) / self.obs_std)
        pf_loss = F.mse_loss(p_pred, r_feats.detach(), reduce=False).sum(1, keepdim=True)
        
        self.pf_std = 0.99 * self.pf_std + 0.01 * pf_loss.std().detach()
        
        int_rewards = self.int_coef * pf_loss.detach() / self.pf_std
        rewards += int_rewards
        
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        q_target = q_target.detach()
        
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())#(((q1_pred - q_target)**2.0) * mask).sum(1).mean()
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())#(((q2_pred - q_target)**2.0) * mask).sum(1).mean()

        """
        VF Loss
        """
        target_qf1 = self.target_qf1(obs, new_actions)
        target_qf2 = self.target_qf2(obs, new_actions)
        minq = torch.min(target_qf1, target_qf2)
        q_new_actions = minq.mean(1, keepdim=True)
        #v_target = q_new_actions - alpha*log_pi
        #vf_loss = 0.5 * self.vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        """
        if self.train_policy_with_reparameterization:
            kl_loss = (alpha*log_pi - q_new_actions).mean()
        else:
            #v_pred = q_new_actions - alpha*log_pi
            log_policy_target = q_new_actions# - v_pred
            kl_loss = (
                log_pi * (alpha*log_pi - log_policy_target).detach()
            ).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = kl_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        
        self.pf_optimizer.zero_grad()
        pf_loss.mean().backward()
        self.pf_optimizer.step()

        #self.vf_optimizer.zero_grad()
        #vf_loss.backward()
        #self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_network()

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            #self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            
            try:
                self.eval_statistics['gmm mus mean'] = np.mean(ptu.get_numpy(self.policy.mean))
                self.eval_statistics['gmm log w mean'] = np.mean(ptu.get_numpy(self.policy.log_w))
                self.eval_statistics['gmm log std mean'] = np.mean(ptu.get_numpy(self.policy.log_std))
            except:
                pass
                
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(kl_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Predictor Loss',
                ptu.get_numpy(pf_loss),
            ))
            self.eval_statistics['Policy Reg Loss'] = np.mean(ptu.get_numpy(
                policy_reg_loss
            ))
            self.eval_statistics['Intrinsic Reward'] = np.mean(ptu.get_numpy(
                int_rewards
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            #self.eval_statistics.update(create_stats_ordered_dict(
            #    'V Predictions',
            #    ptu.get_numpy(v_pred),
            #))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    @property
    def networks(self):
        data = [
            self.policy,
            self.vf,
            self.target_vf,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.pf,
            self.rf,
        ]
        
        return data

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
            qf1=self.qf1,
            qf2=self.qf2,
            rf=self.rf,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            pf=self.pf,
        )
            
        return snapshot
