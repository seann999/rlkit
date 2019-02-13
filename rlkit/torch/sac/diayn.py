from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.core.rl_algorithm import set_to_train_mode, set_to_eval_mode
from rlkit.torch.sac.policies import MakeDeterministic

import gtimer as gt
from rlkit.data_management.path_builder import PathBuilder

from rlkit.policies.base import Policy


class FixedOption(Policy):
    def __init__(self, policy, skill_dim):
        self.policy = policy
        
        vec = np.zeros(skill_dim)
        vec[0] = 1
        self.skill_dim = skill_dim
        self.skill_vec = vec

    def get_action(self, observation):
        observation = np.concatenate([observation, self.skill_vec])
        return self.policy.get_action(observation)

    def get_actions(self, observations):
        return self.policy.get_actions(observations)


class DIAYN(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            qf,
            vf,
            disc,
            skill_dim,

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

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            **kwargs
    ):
        if eval_deterministic:
            eval_policy = FixedOption(MakeDeterministic(policy), skill_dim)
        else:
            eval_policy = FixedOption(policy, skill_dim)
        
        # tell replay buffer about augmented observation space
        env.observation_space.low = np.ones(env.observation_space.shape[0] + skill_dim) * env.observation_space.low[0]
            
        super().__init__(
            env=env,
            # tmp
            training_env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            **kwargs
        )
        
        self.skill_dim = skill_dim
        self.policy = policy
        self.df = disc
        self.qf = qf
        self.vf = vf
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
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.df_criterion = nn.CrossEntropyLoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.df_optimizer = optimizer_class(
            self.df.parameters(),
            lr=policy_lr,
        )
        
        self.sample_z = None
      
    def sample_z_vec(self):
        vec = np.zeros(self.skill_dim)
        vec[np.random.randint(self.skill_dim)] = 1
        
        return vec
    
    # override
    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        
        observation = self._start_new_rollout()
        self.sample_z = self.sample_z_vec()
        observation = np.concatenate([observation, self.sample_z])
        
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):            
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            for t in range(self.num_env_steps_per_epoch):
                #print("step", t, "pool", self.replay_buffer.num_steps_can_sample())
                
                observation = self._take_step_in_env(observation)
                gt.stamp('sample')

                self._try_to_train()
                gt.stamp('train')

            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch(epoch)
            
    def _take_step_in_env(self, observation):
        action, agent_info = self._get_action_and_info(
            observation,
        )
        if self.render:
            self.training_env.render()
        next_ob, raw_reward, terminal, env_info = (
            self.training_env.step(action)
        )
        
        next_ob = np.concatenate([next_ob, self.sample_z])
        
        self._n_env_steps_total += 1
        reward = raw_reward * self.reward_scale
        terminal = np.array([terminal])
        reward = np.array([reward])
        self._handle_step(
            observation,
            action,
            reward,
            next_ob,
            terminal,
            agent_info=agent_info,
            env_info=env_info,
        )
        if terminal or len(self._current_path_builder) >= self.max_path_length:
            self._handle_rollout_ending()
            new_observation = self._start_new_rollout()
            self.sample_z = self.sample_z_vec()
            
            new_observation = np.concatenate([new_observation, self.sample_z])
        else:
            new_observation = next_ob
        return new_observation

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        
        #print(next_obs)

        q_pred = self.qf(obs, actions)
        v_pred = self.vf(obs)
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
            alpha = 0.1
            #alpha = 1
            alpha_loss = 0
            
        
        """
        Discriminator Loss
        """
        actual_obs, skill = obs[:, :-self.skill_dim], obs[:, -self.skill_dim:]
        d_target = skill.argmax(1)
        df_pred = self.df(actual_obs)
        df_loss = self.df_criterion(df_pred, d_target)
        # reuse loss for efficiency?
        log_pred = F.log_softmax(df_pred, 1)[torch.arange(d_target.shape[0]), d_target]
        log_p_z = np.log(1 / self.skill_dim)
        df_rewards = log_pred - log_p_z
        df_rewards = df_rewards.unsqueeze(1).detach()# * 0

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        #q_target = rewards + (1. - terminals) * self.discount * target_v_values
        q_target = df_rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = 0.5 * self.qf_criterion(q_pred, q_target.detach())
        
        scaled_log_pi = alpha*(log_pi - torch.log(1 - new_actions ** 2 + 1e-6).sum(1, keepdim=True))

        """
        VF Loss
        """
        q_new_actions = self.qf(obs, new_actions)
        v_target = q_new_actions - scaled_log_pi#alpha*log_pi
        vf_loss = 0.5 * self.vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        """
        if self.train_policy_with_reparameterization:
            # surrogate kl loss
            policy_loss = (alpha*log_pi - q_new_actions).mean()
        else:
            # surrogate kl loss
            log_policy_target = q_new_actions - v_pred
            policy_loss = (
                log_pi * (scaled_log_pi - log_policy_target).detach()
            ).mean()
        
        mean_reg_loss = self.policy_mean_reg_weight * 0.5 * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * 0.5 * (policy_log_std**2).mean()
        #pre_tanh_value = policy_outputs[-1]
        #pre_activation_reg_loss = self.policy_pre_activation_weight * (
        #    (pre_tanh_value**2).sum(dim=1).mean()
        #)
        
        policy_reg_loss = mean_reg_loss + std_reg_loss# + pre_activation_reg_loss
        kl_loss = policy_loss
        policy_loss = kl_loss + policy_reg_loss

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        
        self.df_optimizer.zero_grad()
        df_loss.backward()
        self.df_optimizer.step()

        self._update_target_network()

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            
            self.eval_statistics['gmm mus mean'] = np.mean(ptu.get_numpy(self.policy.mean))
            self.eval_statistics['gmm log w mean'] = np.mean(ptu.get_numpy(self.policy.log_w))
            self.eval_statistics['gmm log std mean'] = np.mean(ptu.get_numpy(self.policy.log_std))
            
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(
                kl_loss
            ))
            self.eval_statistics['Policy Reg Loss'] = np.mean(ptu.get_numpy(
                policy_reg_loss
            ))
            self.eval_statistics['DF Loss'] = np.mean(ptu.get_numpy(df_loss))
            self.eval_statistics['DF Reward'] = np.mean(ptu.get_numpy(df_rewards))
            self.eval_statistics['DF Log p(z | s)'] = np.mean(ptu.get_numpy(log_pred))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
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
        return [
            self.policy,
            self.qf,
            self.vf,
            self.df,
            self.target_vf,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf=self.qf,
            policy=self.policy,
            vf=self.vf,
            df=self.df,
            target_vf=self.target_vf,
        )
        return snapshot
