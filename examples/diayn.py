import numpy as np
from gym import Env
from gym.spaces import Box

from rlkit.core.serializable import Serializable
from rlkit.envs.wrappers import ProxyEnv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DIAYNWrappedEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            n_skills=6,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        self.sample_z = None
        self.n_skills = n_skills
        
        self.disc = nn.Sequential(
            nn.Linear(self._wrapped_env.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_skills),
        )
        self.disc.cuda()
        self.disc.train()
        self.disc_optim = optim.Adam(self.disc.parameters(), lr=3e-4)

        self.observation_space = Box(
            low=self._wrapped_env.observation_space.low[0],
            high=self._wrapped_env.observation_space.high[0],
            shape=(self._wrapped_env.observation_space.shape[0] + n_skills,))
        
        self.lock_z = False
        

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        
    def reset(self, **kwargs):
        vec = np.zeros(self.n_skills)
        vec[np.random.randint(self.n_skills)] = 1
        
        if not self.lock_z:
            self.sample_z = vec
        
        obs = self._wrapped_env.reset(**kwargs)
        
        return np.concatenate([obs, self.sample_z])
    
    def set_z(self, z_index):
        vec = np.zeros(self.n_skills)
        vec[z_index] = 1
        self.sample_z = vec

    def step(self, action):
        wrapped_step = self._wrapped_env.step(action)
        next_obs, reward, done, info = wrapped_step
        
        # diayn override
        # calculate reward
        current_z_i = np.argmax(self.sample_z)
        q_out = F.log_softmax(self.disc(torch.FloatTensor(next_obs).unsqueeze(0).cuda()), 1)
        log_q_z = q_out[0].cpu().data[current_z_i].item()
        log_p_z = np.log(1 / self.n_skills)
        
        reward = log_q_z - log_p_z
        
        # train discriminator
        q_loss = F.cross_entropy(q_out, torch.LongTensor([current_z_i]).cuda())
        self.disc_optim.zero_grad()
        q_loss.backward()
        self.disc_optim.step()
        
        # augment observation with skill vector
        next_obs = np.concatenate([next_obs, self.sample_z])

        return next_obs, reward, done, info
    
    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["_diayn_disc"] = self.disc.state_dict()
        d["_diayn_optim"] = self.disc_optim.state_dict()
        d["_n_skills"] = self.n_skills
        d["_sample_z"] = self.sample_z
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.disc.load_state_dict(d["_diayn_disc"])
        self.disc_optim.load_state_dict(d["_diayn_optim"])
        self.n_skills = d["_n_skills"]
        self.sample_z = d["_sample_z"]

    def __str__(self):
        return "DIAYN: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)