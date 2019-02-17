from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn
import gtimer as gt
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.sac.policies import MultiMakeDeterministic
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core.rl_algorithm import set_to_train_mode, set_to_eval_mode
from rlkit.torch.sac.rndsac import CustomReplayBuffer, get_dim

import torch.multiprocessing as mp

class ThompsonSoftActorCritic(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policies,
            qf1s,
            qf2s,
            pqf1s,
            pqf2s,

            policy_lr=1e-3,
            qf_lr=1e-3,
            droprate=0.5,
            prior_coef=1,
            prior_offset=0,
            heads=10,
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
            eval_policy = MultiMakeDeterministic(policies[0])
        else:
            eval_policy = policies[0]
            
        self.heads = heads
        self.prior_coef = prior_coef
        replay_buffer = CustomReplayBuffer(
            replay_buffer_size,
            env,
            get_dim(env.observation_space) + self.heads
        )
            
        super().__init__(
            env=env,
            exploration_policy=policies[0],
            eval_policy=eval_policy,
            replay_buffer=replay_buffer,
            **kwargs
        )
        self.current_behavior_policy = 0
        self.policies = policies
        self.qf1s = qf1s
        self.qf2s = qf2s
        self.pqf1s = pqf1s
        self.pqf2s = pqf2s
        self.droprate = droprate

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
            
            self.log_alphas = [ptu.zeros(1, requires_grad=True) for _ in range(heads)]
            self.alpha_optimizers = [optimizer_class(
                [log_alpha],
                lr=policy_lr,
            ) for log_alpha in self.log_alphas]
            
        self.prior_offset = prior_offset

        self.target_qf1s = [qf1.copy() for qf1 in qf1s]
        self.target_qf2s = [qf2.copy() for qf2 in qf2s]
        self.qf_criterion = nn.MSELoss()

        self.policy_optimizers = [optimizer_class(
            policy.parameters(),
            lr=policy_lr,
        ) for policy in policies]
        self.qf1_optimizers = [optimizer_class(
            qf1.parameters(),
            lr=qf_lr,
        ) for qf1 in qf1s]
        self.qf2_optimizers = [optimizer_class(
            qf2.parameters(),
            lr=qf_lr,
        ) for qf2 in qf2s]

    def train_head(self, head):
                
        for step in range(1, 1000):
            if step % 100 == 0:
                print(head, step)
            self._do_training(head)
        
    def _do_training(self, head):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations'][:, :-self.heads]
        mask = batch['observations'][:, -self.heads + head]
        actions = batch['actions']
        next_obs = batch['next_observations'][:, :-self.heads]
        
        qf1 = self.qf1s[head]
        qf2 = self.qf2s[head]
        pqf1 = self.pqf1s[head]
        pqf2 = self.pqf2s[head]
        
        target_qf1 = self.target_qf1s[head]
        target_qf2 = self.target_qf2s[head]
        policy = self.policies[head]

        q1_pred = qf1(obs, actions) + self.prior_coef * pqf1(obs, actions) + self.prior_offset
        q2_pred = qf2(obs, actions) + self.prior_coef * pqf2(obs, actions) + self.prior_offset
        
        qf1_loss, qf2_loss = 0, 0

        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = policy(
            obs,
            reparameterize=self.train_policy_with_reparameterization,
            return_log_prob=True,
        )
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        
        # new_actions: 128 x 10 x 1

        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alphas[head] * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizers[head].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizers[head].step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """
        QF Loss
        """
        next_policy_outputs = policy(
            next_obs,
            reparameterize=self.train_policy_with_reparameterization,
            return_log_prob=True,
        )
        next_new_actions, _, _, next_log_pi = next_policy_outputs[:4]
<<<<<<< HEAD

        next_q_new_actions = torch.min(
            target_qf1(next_obs, next_new_actions) + self.prior_coef * pqf1(next_obs, next_new_actions) + self.prior_offset,
            target_qf2(next_obs, next_new_actions) + self.prior_coef * pqf2(next_obs, next_new_actions) + self.prior_offset,
        )
=======
        # 128 x 10
        
        def get_q(qf1, qf2, obs, actions):
            # actions: 128 x 10 x 3
            
            # 1280 x 3
            flat_actions = actions.view(-1, actions.shape[2])
            
            # 128 x 5 -> 1280 x 5
            expand_obs = obs.repeat(1, self.heads).view(-1, obs.shape[1])
            
            # 1280 x 10
            q = torch.min(
                self.target_qf1(expand_obs, flat_actions) + self.prior_coef * self.pqf1(expand_obs, flat_actions) + self.prior_offset,
                self.target_qf2(expand_obs, flat_actions) + self.prior_coef * self.pqf2(expand_obs, flat_actions) + self.prior_offset,
            )
            
            # 128 x 10 x 10
            q = q.view(-1, self.heads, self.heads)
            q = q[:, torch.arange(self.heads), torch.arange(self.heads)]
            
            return q
        
        # 128 x 10
        next_q_new_actions = get_q(self.target_qf1, self.target_qf2, next_obs, next_new_actions)
>>>>>>> parent of 3037648... synced thompsac
        
        # 128 x 10
        target_v_values = next_q_new_actions - alpha*next_log_pi

        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf1_loss = (((q1_pred - q_target.detach())**2.0) * mask).sum(1)
        qf2_loss = (((q2_pred - q_target.detach())**2.0) * mask).sum(1)

        """
        VF Loss
        """
        q_new_actions = torch.min(
            qf1(obs, new_actions) + self.prior_coef * pqf1(obs, new_actions) + self.prior_offset,
            qf2(obs, new_actions) + self.prior_coef * pqf2(obs, new_actions) + self.prior_offset,
        )

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
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).sum(1).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).sum(1).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = kl_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf1_optimizers[head].zero_grad()
        qf1_loss.mean().backward()
        self.qf1_optimizers[head].step()

        self.qf2_optimizers[head].zero_grad()
        qf2_loss.mean().backward()
        self.qf2_optimizers[head].step()

        self.policy_optimizers[head].zero_grad()
        policy_loss.backward()
        self.policy_optimizers[head].step()

        self._update_target_network(head)

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
                
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(
                kl_loss
            ))
            self.eval_statistics['Policy Reg Loss'] = np.mean(ptu.get_numpy(
                policy_reg_loss
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
                
    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            
            self.training_mode(True)
            processes = []
            
            import threading
            gt.stamp('sample')
            
            if self._can_train():
                ctx = mp.get_context("spawn")
                
                for net in self.networks:
                #    net.cuda()
                    net.share_memory()
                
                for rank in range(0, self.heads):
                    p = ctx.Process(target=self.train_head, args=(rank,))
                    p.start()
                    processes.append(p)
            
            for step in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
                #self._try_to_train()
                
            for p in processes:
                p.join()
            gt.stamp('train')
            self.training_mode(False)
            self._n_train_steps_total += self.num_env_steps_per_epoch

            self.current_behavior_policy = np.random.randint(self.heads)
            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch(epoch)
                
    def _take_step_in_env(self, observation):
        #action, agent_info = self._get_action_and_info(
        #    observation,
        #)
        self.policies[0].set_num_steps_total(self._n_env_steps_total)
        action, agent_info = self.policies[0].get_action(
            observation,
            self.current_behavior_policy,
        )
        
        if self.render:
            self.training_env.render()
        next_ob, raw_reward, terminal, env_info = (
            self.training_env.step(action)
        )
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
        else:
            new_observation = next_ob
        return new_observation
                
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
            mask = np.array([np.random.uniform() > self.droprate for _ in range(self.heads)])#np.random.randint(2, size=self.heads)
            #mask = [1] * self.heads

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

    @property
    def networks(self):
        nets = []
        nets.extend(self.qf1s)
        nets.extend(self.qf2s)
        nets.extend(self.pqf1s)
        nets.extend(self.pqf2s)
        nets.extend(self.target_qf1s)
        nets.extend(self.target_qf2s)
        nets.extend(self.policies)
        return nets

    def _update_target_network(self, head):
        ptu.soft_update_from_to(self.qf1s[head], self.target_qf1s[head], self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2s[head], self.target_qf2s[head], self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        
        for i, (qf1, qf2, pqf1, pqf2, target_qf1, target_qf2, policy) in enumerate(zip(self.qf1s, self.qf2s, self.pqf1s, self.pqf2s, self.target_qf1s, self.target_qf2s, self.policies)):
            snapshot["qf1_{}".format(i)] = qf1
            snapshot["qf2_{}".format(i)] = qf2
            snapshot["pqf1_{}".format(i)] = pqf1
            snapshot["pqf2_{}".format(i)] = pqf2
            snapshot["target_qf1_{}".format(i)] = target_qf1
            snapshot["target_qf2_{}".format(i)] = target_qf2
            snapshot["policy_{}".format(i)] = policy

        return snapshot
