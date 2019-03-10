from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import gtimer as gt
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.sac.policies import MultiMakeDeterministic
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core.rl_algorithm import set_to_train_mode, set_to_eval_mode
from rlkit.torch.sac.rndsac import CustomReplayBuffer, get_dim
from rlkit.core import eval_util, logger

from rlkit.torch.networks import FlattenMlp, SplitFlattenMlp, EnsembleFlattenMlp

class ThompsonSoftActorCritic(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            policyB,
            policyC,
            qf1,
            qf2,
            qfB,
            pqf1,
            pqf2,

            policy_lr=1e-3,
            qf_lr=1e-3,
            droprate=0.5,
            prior_coef=1,
            heads=10,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            alpha=1,
            alphaC=1,
            int_w = 0.1,
            int_discount=0.99,
            int_direct=False,
            rnd=False,
            newmethod=False,

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
        if newmethod:
            if eval_deterministic:
                eval_policy = MultiMakeDeterministic(policyC)
            else:
                eval_policy = policyC
        else:
            if eval_deterministic:
                eval_policy = MultiMakeDeterministic(policy)
            else:
                eval_policy = policy
            
        self.heads = heads
        self.prior_coef = prior_coef
        replay_buffer = CustomReplayBuffer(
            replay_buffer_size,
            env,
            get_dim(env.observation_space) + self.heads + 1
        )
            
        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            replay_buffer=replay_buffer,
            **kwargs
        )
        self.current_behavior_policy = 0
        self.policy = policy
        self.policyB = policyB
        self.policyC = policyC
        self.alpha = alpha
        self.alphaC = alphaC
        self.int_w = int_w
        self.int_discount = int_discount
        self.int_direct = int_direct
        self.rnd = rnd
        self.newmethod = newmethod
        self.qf1 = qf1
        self.qf2 = qf2
        self.pqf1 = pqf1
        self.pqf2 = pqf2
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
            
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
            
        self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            
        self.log_alphaC = ptu.zeros(1, requires_grad=True)
        self.alpha_optimizerC = optimizer_class(
            [self.log_alphaC],
            lr=policy_lr,
        )
            
        self.qfB = qfB
        self.target_qfB = self.qfB.copy()
        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()

        self.policy_optimizer = optimizer_class(
            policy.parameters(),
            lr=policy_lr,
        )
        self.policyB_optimizer = optimizer_class(
            self.policyB.parameters(),
            lr=policy_lr,
        )
        self.policyC_optimizer = optimizer_class(
            self.policyC.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.qfB_optimizer = optimizer_class(
            self.qfB.parameters(),
            lr=qf_lr,
        )
        
        self.counts = np.zeros(100)
        self.targets = []
        self.targets2 = []
        
    def get_q(self, obs, actions, qf1, qf2=None, pqf1=None, pqf2=None):
        # actions: N x K x A
        
        K = actions.shape[1]
        A = actions.shape[2]
        S = obs.shape[1]
        
        if type(qf1) == FlattenMlp:
            # NK x A
            flat_actions = actions.view(-1, A)

            # N x S -> NK x S
            expand_obs = obs.repeat(1, K).view(-1, S)

            if qf2:
                # NK x K
                q = torch.min(
                    qf1(expand_obs, flat_actions) + self.prior_coef * pqf1(expand_obs, flat_actions),
                    qf2(expand_obs, flat_actions) + self.prior_coef * pqf2(expand_obs, flat_actions),
                )
            elif pqf1:
                q = qf1(expand_obs, flat_actions) + self.prior_coef * pqf1(expand_obs, flat_actions)
            else:
                q = qf1(expand_obs, flat_actions)

            # N x K x K
            q = q.view(-1, K, K)
            q = q[:, torch.arange(K), torch.arange(K)]

            return q
        elif type(qf1) == SplitFlattenMlp or type(qf1) == EnsembleFlattenMlp:
            inp = [[obs, actions[:, i, :]] for i in range(K)]

            if qf2:
                q = torch.min(
                    qf1(inp) + self.prior_coef * pqf1(inp),
                    qf2(inp) + self.prior_coef * pqf2(inp),
                )
            elif pqf1:
                q = qf1(inp) + self.prior_coef * pqf1(inp)
            else:
                q = qf1(inp)

            return q

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations'][:, :-self.heads-1]
        mask = batch['observations'][:, -self.heads-1:-1]
        pol = batch['observations'][:, -1]
        actions = batch['actions']
        next_obs = batch['next_observations'][:, :-self.heads-1]

        q1_pred = self.qf1(obs, actions) + self.prior_coef * self.pqf1(obs, actions)
        q2_pred = self.qf2(obs, actions) + self.prior_coef * self.pqf2(obs, actions)
        
        qf1_loss, qf2_loss = 0, 0

        """
        Calculate log pi & other policy vars
        """
        policy_outputs = self.policy(
            obs,
            reparameterize=self.train_policy_with_reparameterization,
            return_log_prob=True,
        )
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        Update alpha
        """
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha
            alpha_loss = 0

        """
        Update Q Ensemble
        """
        next_policy_outputs = self.policy(
            next_obs,
            reparameterize=self.train_policy_with_reparameterization,
            return_log_prob=True,
        )
        next_new_actions, _, _, next_log_pi = next_policy_outputs[:4]
        
        # N x K
        next_q_new_actions = self.get_q(next_obs, next_new_actions, self.target_qf1, self.target_qf2, self.pqf1, self.pqf2)
        
        # N x K
        entropy_bonus = -alpha * next_log_pi[:, torch.arange(self.heads), 0]
        target_v_values = next_q_new_actions + entropy_bonus

        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        
        if self.rnd:
            q_target *= 0
            
        qf1_loss = (((q1_pred - q_target.detach())**2.0) * mask).sum(1)
        qf2_loss = (((q2_pred - q_target.detach())**2.0) * mask).sum(1)
        
        self.targets.append(np.hstack([obs.detach().cpu().numpy(), q_target.detach().cpu().numpy(), mask.detach().cpu().numpy(), ((1. - terminals) * self.discount * entropy_bonus).detach().cpu().numpy(), pol.cpu().numpy()[:, None]]))
        
        """
        Update ensemble maximizers
        """
        q_new_actions = self.get_q(obs, new_actions, self.qf1, self.qf2, self.pqf1, self.pqf2)
        
        kl_loss = (alpha * log_pi[:, torch.arange(self.heads), 0] - q_new_actions).sum(1).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).sum(1).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).sum(1).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = kl_loss + policy_reg_loss
        
        """
        Calculate log pi & other policy vars (2)
        """
        # N x 2
        # one head for greedy, other for curious
        policy_outputsB = self.policyB(
            obs,
            reparameterize=self.train_policy_with_reparameterization,
            return_log_prob=True,
        )
        new_actionsB, policy_meanB, policy_log_stdB, log_piB = policy_outputsB[:4]
        
        alpha_lossC = -(self.log_alphaC * (log_piB + self.target_entropy).detach()).mean()
        self.alpha_optimizerC.zero_grad()
        alpha_lossC.backward()
        self.alpha_optimizerC.step()
        alphaB = self.log_alphaC.exp()
        
        """
        Update Q B
        """
        # calculate intrinsic rewards (Nx1)
        if self.rnd:
            int_rewards = (q1_pred.detach()**2.0).sum(1).unsqueeze(1)
        else:
            int_rewards = torch.min(q1_pred.detach(), q2_pred.detach()).std(1).unsqueeze(1)
        
        # N x 2
        # calc maximum
        next_policy_outputsB = self.policyB(
            next_obs,
            reparameterize=self.train_policy_with_reparameterization,
            return_log_prob=True,
        )
        next_new_actionsB, _, _, next_log_piB = next_policy_outputsB[:4]
        
        # N x 2
        next_q_new_actionsB = self.get_q(next_obs, next_new_actionsB, self.target_qfB)
        next_QE = next_q_new_actionsB[:, [0]]
        next_QI = next_q_new_actionsB[:, [1]]
        
        alphaB = 0
        entropy_bonusB = -alphaB * next_log_piB[:, [1], 0]
        # extrinsic target value
        qB_target_E = rewards + (1. - terminals) * self.discount * (next_QE + entropy_bonusB)
        self.targets2.append(np.hstack([obs.detach().cpu().numpy(), qB_target_E.detach().cpu().numpy(), actions.detach().cpu().numpy()]))
        # extrinsic + intrinsic target value
        qB_target_I = (rewards + self.int_w * int_rewards) + (1. - terminals) * self.int_discount * (next_QI + entropy_bonusB)
        qB_target = torch.cat([qB_target_E, qB_target_I], 1)
        
        qB_pred = self.qfB(obs, actions)
        qB_loss = ((qB_pred - qB_target.detach())**2.0).sum(1)
        
        """
        Update policy B
        """
        q_new_actionsB = self.get_q(obs, new_actionsB, self.qfB)
        
        kl_lossB = (alphaB * log_piB[:, torch.arange(2), 0] - q_new_actionsB).sum(1).mean()

        mean_reg_lossB = self.policy_mean_reg_weight * (policy_meanB**2).sum(1).mean()
        std_reg_lossB = self.policy_std_reg_weight * (policy_log_stdB**2).sum(1).mean()
        pre_tanh_valueB = policy_outputsB[-1]
        pre_activation_reg_lossB = self.policy_pre_activation_weight * (
            (pre_tanh_valueB**2).sum(dim=1).mean()
        )
        policy_reg_lossB = mean_reg_lossB + std_reg_lossB + pre_activation_reg_lossB
        policy_lossB = kl_lossB + policy_reg_lossB
        
        """
        Update policy C
        """
#         policy_outputsC = self.policyC(
#             obs,
#             reparameterize=self.train_policy_with_reparameterization,
#             return_log_prob=True,
#         )
#         new_actionsC, policy_meanC, policy_log_stdC, log_piC = policy_outputsC[:4]
        
#         alpha_lossC = -(self.log_alphaC * (log_piC + self.target_entropy).detach()).mean()
#         self.alpha_optimizerC.zero_grad()
#         alpha_lossC.backward()
#         self.alpha_optimizerC.step()
#         alphaC = self.log_alphaC.exp()
        
#         q_new_actionsC = self.qfB(obs, new_actionsC[:, 0, :])
#         q_new_actionsC = q_new_actionsC[:, 0] + self.int_w * q_new_actionsC[:, 1]
        
#         kl_lossC = (alphaC * log_piC[:, 0, 0] - q_new_actionsC).mean()

#         mean_reg_lossC = self.policy_mean_reg_weight * (policy_meanC**2).sum(1).mean()
#         std_reg_lossC = self.policy_std_reg_weight * (policy_log_stdC**2).sum(1).mean()
#         pre_tanh_valueC = policy_outputsC[-1]
#         pre_activation_reg_lossC = self.policy_pre_activation_weight * (
#             (pre_tanh_valueC**2).sum(dim=1).mean()
#         )
#         policy_reg_lossC = mean_reg_lossC + std_reg_lossC + pre_activation_reg_lossC
#         policy_lossC = kl_lossC + policy_reg_lossC

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.mean().backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.mean().backward()
        self.qf2_optimizer.step()
        
        self.qfB_optimizer.zero_grad()
        qB_loss.mean().backward()
        self.qfB_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.policyB_optimizer.zero_grad()
        policy_lossB.backward()
        self.policyB_optimizer.step()
        
        # self.policyC_optimizer.zero_grad()
        # policy_lossC.backward()
        # self.policyC_optimizer.step()
        
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
                
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(
                kl_loss
            ))
            self.eval_statistics['Policy Reg Loss'] = np.mean(ptu.get_numpy(
                policy_reg_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Intrinsic Reward',
                ptu.get_numpy(int_rewards),
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
            self.targets = []
            self.targets2 = []
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            for step in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
                gt.stamp('sample')

                self._try_to_train()
                gt.stamp('train')

            self.current_behavior_policy = np.random.randint(self.heads)
            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            
            if epoch % 10 == 0:
                import matplotlib.pyplot as plt
                import traceback
                
                cc = plt.rcParams['axes.color_cycle']
                
                plt.clf()
                fig, axes = plt.subplots(1, 5)
                fig.set_size_inches(24, 6)
                ax1 = axes[0]
                #ax1.set_ylim(-1, 10)
                #¥ax2 = ax1.twinx()
                
                graph_x = np.arange(0, 100)
                try:
                    x = np.array([self.training_env.preprocess(xx) for xx in graph_x])
                    obs = torch.from_numpy(x).float().cuda()# / 100
                    acts = torch.from_numpy(np.ones((100, 1)) * self.env.mask[:, None]).float().cuda()
                    acts2 = torch.from_numpy(np.ones((100, 1))*-1 * self.env.mask[:, None]).float().cuda()
                    p1 = self.prior_coef * self.pqf1(obs, acts)
                    p2 = self.prior_coef * self.pqf2(obs, acts)
                    qf11 = self.qf1(obs, acts)
                    qf21 = self.qf2(obs, acts)

                    q, qi1 = torch.min(torch.stack([qf11+p1, qf21+p2], 2), 2)

                    p3 = self.prior_coef * self.pqf1(obs, acts2)
                    p4 = self.prior_coef * self.pqf2(obs, acts2)
                    qf12 = self.qf1(obs, acts2)
                    qf22 = self.qf2(obs, acts2)
                    q2, qi2 = torch.min(torch.stack([qf12+p3, qf22+p4], 2), 2)

                    #p = torch.stack([p1, p2], 2).view(-1, 2)
                    #p = p[np.arange(p.shape[0]), qi1.flatten()].view(-1, self.heads)
                    #p2 = torch.stack([p3, p4], 2).view(-1, 2)
                    #p2 = p2[np.arange(p2.shape[0]), qi2.flatten()].view(-1, self.heads)

                    q = q.cpu().detach().numpy()
                    q2 = q2.cpu().detach().numpy()

                    ax1.plot(graph_x, p1.cpu().detach().numpy(), c="red", ls="-.", lw=1)
                    ax1.plot(graph_x, p2.cpu().detach().numpy(), c="red", ls="-.", lw=1)
                    ax1.plot(graph_x, p3.cpu().detach().numpy(), c="red", ls=":", lw=1)
                    ax1.plot(graph_x, p4.cpu().detach().numpy(), c="red", ls=":", lw=1)

                    for i in range(self.heads):
                        ax1.plot(graph_x, q[:, i], c=cc[i % len(cc)], lw=3)
                        ax1.plot(graph_x, q2[:, i], ls="--", c=cc[i % len(cc)], lw=3)

                    ax1.set_ylim(min(q.min(), q2.min()), max(q.max(), q2.max()))

                    if len(self.targets) > 0:
                        self.targets = np.vstack(self.targets)
                        axes[1].axhline(0, color='black')
                        axes[1].axvline(0, color='black')

                        for i in range(self.heads):
                            for k in range(self.heads):
                                # obs target(H) mask(H) entropybonus(H) policy(1)
                                m = np.all([self.targets[:, -1] == i, self.targets[:, x.shape[1]+self.heads+k] == 1], 0)
                                axes[1].scatter(self.targets[m, :x.shape[1]].argmax(1), self.targets[m, x.shape[1]+k], s=10, linewidths=0.5, alpha=0.5, c=cc[k % len(cc)], #[i]
                                               edgecolors=cc[k % len(cc)])
                                axes[1].scatter(self.targets[m, :x.shape[1]].argmax(1), self.targets[m, x.shape[1]+self.heads*2+k], s=10, linewidths=0.5, alpha=0.5, c=cc[k % len(cc)], #[i]
                                               edgecolors=cc[k % len(cc)], marker="x")

                    ymin, ymax = axes[0].get_ylim()
                    axes[1].set_ylim(ymin, ymax)
                    xmin, xmax = axes[0].get_xlim()
                    axes[1].set_xlim(xmin, xmax)
                    
                    ax22 = axes[1].twinx()
                    ax22.semilogy(range(100), self.counts)

                    #ax1.plot([0, 100], [R, R], ls="--")

                    #ax2.semilogy(range(100), self.counts)
                    axes[0].axvline(0, color='black')

                    policy_outputs = self.policy(
                        obs,
                        reparameterize=self.train_policy_with_reparameterization,
                        return_log_prob=False,
                    )
                    next_new_actions, policy_mean, policy_log_std, next_log_pi = policy_outputs[:4]
                    mask_cuda = torch.FloatTensor(self.env.mask).cuda()
                    pt_mean = F.tanh(policy_mean) * mask_cuda[:, None, None]

                    policy_std = policy_log_std.exp()
                    policy_low = F.tanh(policy_mean - policy_std).detach().cpu().numpy()[..., 0]
                    policy_high = F.tanh(policy_mean + policy_std).detach().cpu().numpy()[..., 0]

                    policy_mean = pt_mean.detach().cpu().numpy()

                    axes[2].set_ylim(-1.1, 1.1)

                    mean = policy_mean[:, :, 0]
                    std = policy_std[:, :, 0]

                    axes[2].axhline(0, color='black')
                    axes[2].axvline(0, color='black')
                    
                    correct = (q > q2) == (mean < 0)

                    for i in range(self.heads):
                        axes[2].fill_between(graph_x, policy_low[:, i], policy_high[:, i], color=cc[i % len(cc)] + "20")
                        
                        #left = q[:, i] > q2[:, i]
                        #axes[2].scatter(graph_x[left], mean[left, i], color=plt.rcParams['axes.color_cycle'][i], marker="+")
                        
                    for i in range(self.heads):
                        axes[2].scatter(graph_x[correct[:, i]], mean[correct[:, i], i], color="red")
                        axes[2].scatter(graph_x[np.invert(correct[:, i])], mean[np.invert(correct[:, i]), i], color="blue")

                    p = axes[2].plot(graph_x, mean, lw=1)

                    axes[3].set_ylim(ymin, ymax)
                    mean1 = q.mean(1)
                    mean2 = q2.mean(1)
                    std1 = q.std(1)
                    std2 = q2.std(1)
                    
                    plot1 = axes[3].plot(graph_x, mean1, label="right")
                    axes[3].plot(graph_x, std1, color=plot1[0].get_color(), ls="--")
                    plot2 = axes[3].plot(graph_x, mean2, label="left")
                    axes[3].plot(graph_x, std2, color=plot2[0].get_color(), ls="--")
                    axes[3].fill_between(graph_x, mean1-std1, mean1+std1, color=plot1[0].get_color() + "20")
                    axes[3].fill_between(graph_x, mean2-std2, mean2+std2, color=plot2[0].get_color() + "20")

                    ax32 = axes[3].twinx()
                    
                    if len(self.targets2) > 0:
                        self.targets2 = np.vstack(self.targets2)
                        move = self.targets2[:, x.shape[1]+1]
                        
                        left = move < 0
                        ax32.scatter(self.targets2[left, :x.shape[1]].argmax(1), self.targets2[left, x.shape[1]], s=5, c=move[left], cmap="Reds")
                        right = move >= 0
                        ax32.scatter(self.targets2[right, :x.shape[1]].argmax(1), self.targets2[right, x.shape[1]], s=5, c=move[right], cmap="Blues")
                        
                    qf31 = self.qf3(obs, acts).cpu().detach().numpy()[:, 0]
                    qf32 = self.qf3(obs, acts2).cpu().detach().numpy()[:, 0]
                    ax32.plot(graph_x, qf31, label="right", ls=":")
                    ax32.plot(graph_x, qf32, label="left", ls=":")
                    axes[3].legend()
                    
                    policy_outputs = self.policy2(
                        obs,
                        reparameterize=self.train_policy_with_reparameterization,
                        return_log_prob=False,
                    )
                    next_new_actions, policy_mean, policy_log_std, next_log_pi = policy_outputs[:4]
                    pt_mean = F.tanh(policy_mean) * mask_cuda[:, None, None]
                    
                    policy_std = policy_log_std.exp()
                    policy_low = (F.tanh(policy_mean - policy_std) * mask_cuda[:, None, None]).detach().cpu().numpy()[:, 0, 0]
                    policy_high = (F.tanh(policy_mean + policy_std) * mask_cuda[:, None, None]).detach().cpu().numpy()[:, 0, 0]
                    
                    policy_mean = pt_mean.detach().cpu().numpy()[:, 0, 0]
                    axes[4].set_ylim(-1, 1)
                    plot3 = axes[4].plot(graph_x, policy_mean)
                    axes[4].fill_between(graph_x, policy_low, policy_high, color=plot3[0].get_color() + "20")

                    # for i in range(self.heads):
                    #     maxact = q[:, i].argmax()
                    #     axes[1].scatter(x[maxact], mean[maxact, i])

                    plt.tight_layout()
                    path = "{}/Q-{:04d}.png".format(logger._snapshot_dir, epoch)
                    plt.savefig(path)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
            
            self._end_epoch(epoch)
                
    def _take_step_in_env(self, observation):
        #action, agent_info = self._get_action_and_info(
        #    observation,
        #)
        self.policy.set_num_steps_total(self._n_env_steps_total)
        
        if self.newmethod:
            action, agent_info = self.policyB.get_action(
                observation,
                1,
            )
        else:
            action, agent_info = self.policy.get_action(
                observation,
                self.current_behavior_policy,
            )
        
        if self.render:
            self.training_env.render()

        next_ob, raw_reward, terminal, env_info = (
            self.training_env.step(action)
        )
        
        #x = int(next_ob[0])
        x = int(next_ob.argmax())
        if x >= 0 and x < len(self.counts):
            self.counts[x] += 1
        
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

        observation = np.concatenate([observation, mask, [self.current_behavior_policy]])
        next_observation = np.concatenate([next_observation, np.zeros(self.heads), [self.current_behavior_policy]])
        
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
        nets = [
            self.qf1,
            self.qf2,
            self.qfB,
            self.target_qf1,
            self.target_qf2,
            self.target_qfB,
            self.pqf1,
            self.pqf2,
            self.policy,
            self.policyB,
            self.policyC,
        ]
        return nets

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        ptu.soft_update_from_to(self.qfB, self.target_qfB, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf1=self.qf1,
            qf2=self.qf2,
            qfB=self.qfB,
            policy=self.policy,
            policyB=self.policyB,
            policyC=self.policyC,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            target_qfB=self.target_qfB,
            pqf1=self.pqf1,
            pqf2=self.pqf2,
        )
        return snapshot
