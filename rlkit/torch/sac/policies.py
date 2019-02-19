import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import SplitMlp, Mlp

from torch.distributions import Categorical, Normal


#LOG_SIG_MAX = 2
#LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
LOG_W_MIN = -10


class SplitMultiTanhGaussianPolicy(SplitMlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            heads,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            heads=heads,
            init_w=init_w,
            **kwargs
        )

        self.log_std = None
        self.std = std
        self.heads = heads
        self.action_dim = action_dim
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std =  nn.Conv1d(last_hidden_size * heads, action_dim * heads, 1, groups=heads)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, policy_i, deterministic=False):
        actions = self.get_actions(obs_np[None], policy_i, deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, policy_i, deterministic=False):
        actions = self.eval_np(obs_np, deterministic=deterministic)[0]
        return actions[:, policy_i]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs.unsqueeze(2).repeat(1, self.heads, 1)
        for i, fc in enumerate(self.convs):
            h = self.hidden_activation(fc(h))
        mean = self.last_conv(h).squeeze(2) # actions * heads

        if self.std is None:
            log_std = self.last_fc_log_std(h).squeeze(2)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        log_stds = None
        log_probs = None
        
        if deterministic:
            means = mean.view(-1, self.action_dim)
            actions = torch.tanh(means)
            actions = actions.view(-1, self.heads, self.action_dim)
        else:
            all_actions = []
            means = mean.view(-1, self.action_dim)
            stds = std.view(-1, self.action_dim)
            log_stds = log_std.view(-1, self.action_dim)
            
            tanh_normal = TanhNormal(means, stds)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
                
                log_probs = log_prob.view(-1, self.heads, self.action_dim)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

            actions = action.view(-1, self.heads, self.action_dim)
                
        return (
            actions, means, log_stds, log_probs, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )
    
class MultiTanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            heads,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * heads,
            init_w=init_w,
            **kwargs
        )

        self.log_std = None
        self.std = std
        self.heads = heads
        self.action_dim = action_dim
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim * heads)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, policy_i, deterministic=False):
        actions = self.get_actions(obs_np[None], policy_i, deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, policy_i, deterministic=False):
        actions = self.eval_np(obs_np, deterministic=deterministic)[0]
        return actions[:, policy_i]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h) # actions * heads
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        log_stds = None
        log_probs = None
        
        if deterministic:
            means = mean.view(-1, self.action_dim)
            actions = torch.tanh(means)
            actions = actions.view(-1, self.heads, self.action_dim)
        else:
            all_actions = []
            means = mean.view(-1, self.action_dim)
            stds = std.view(-1, self.action_dim)
            log_stds = log_std.view(-1, self.action_dim)
            
            tanh_normal = TanhNormal(means, stds)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
                
                log_probs = log_prob.view(-1, self.heads, self.action_dim)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

            actions = action.view(-1, self.heads, self.action_dim)
                
        return (
            actions, means, log_stds, log_probs, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        actions = self.eval_np(obs_np, deterministic=deterministic)[0]
        return actions

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )
    
from torch.distributions import MultivariateNormal
    
def log_gaussian(mu_t, log_std_t, x):
    normalized_dist_t = (x - mu_t) * torch.exp(-log_std_t)
    quadratic = - 0.5 * (normalized_dist_t ** 2).sum(2)

    log_z = log_std_t.sum(2)
    D_t = mu_t.shape[2]
    log_z += 0.5 * D_t * np.log(2 * np.pi)

    log_p = quadratic - log_z

    return log_p
    
    
class GMMPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            hidden_init=torch.nn.init.xavier_uniform_,
            k=2,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_stds = None
        self.k = k
        self.action_dim = action_dim

        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
            
        self.last_fc = nn.Linear(last_hidden_size, k * (2 * action_dim + 1))
        hidden_init(self.last_fc.weight)
        self.last_fc.bias.data.fill_(0)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        out = self.last_fc(h).view(-1, self.k, (2 * self.action_dim + 1))

        log_w = out[..., 0]
        mean = out[..., 1:1+self.action_dim]
        log_std = out[..., 1+self.action_dim:]
        
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        log_w = torch.clamp(log_w, min=LOG_W_MIN)
        
        self.log_w = log_w
        self.mean = mean
        self.log_std = log_std
        
        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        
        arange = torch.arange(out.shape[0])
        
        if deterministic:
            ks = log_w.view(-1, self.k).argmax(1)
            action = torch.tanh(mean[arange, ks])
        else:
            sample_ks = Categorical(logits=log_w.view(-1, self.k)).sample()
            
            tanh_normal = TanhNormal(mean[arange, sample_ks], std[arange, sample_ks])
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )

                # (NxKxA), (NxKxA), (Nx1xA) => (NxK)
                log_p_xz_t = log_gaussian(mean, log_std, pre_tanh_value[:, None, :].data)

                log_p_x_t = torch.logsumexp(log_p_xz_t + log_w, 1) - torch.logsumexp(log_w, 1)
                
                # squash correction
                log_prob = log_p_x_t - torch.log(1 - action ** 2 + 1e-6).sum(1)

                log_prob = log_prob[:, None]
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )    
    
    
class MultiMakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation, 0,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations, 0,
                                                  deterministic=True)


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
