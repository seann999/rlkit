import gym
import numpy as np

from garage.core import Serializable
from garage.envs import Step

import pickle
import os

class LineEnv(gym.Env, Serializable):
    
    def __init__(
            self,
    ):
        self.pos = 1
        self.steps = 0
        if not os.path.exists("mask.p"):
            pickle.dump(np.random.randint(0, 2, size=100) * 2 - 1, open("mask.p", "wb"))
        self.mask = pickle.load(open("mask.p", "rb"))
        #self.mask = [1]*50
        #self.mask.extend([-1]*50)
        #self.mask = np.array(self.mask)
        Serializable.quick_init(self, locals())

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(100, ), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1, high=1, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.pos = np.random.uniform() * 2 - 1
        self.steps = 0
        return self.preprocess(self.pos)

    def step(self, action):
        a = action.copy()[0] * self.mask[int(min(max(self.pos, 0), len(self.mask)-1))]
        a = np.clip(a, self.action_space.low, self.action_space.high)[0]

        self.pos += a

        if self.pos > 90:
            reward = 1
            done = False
        elif a > 0:
            reward = -0.01
            done = False
        else:
            done = False
            reward = 0

        self.steps += 1
        
        if self.steps >= 100:
            done = True

        return self.preprocess(self.pos), reward, done, None
    
    def preprocess(self, obs):
        vec = np.zeros(100)
        vec[min(max(0, int(obs)), 99)] = 1
        #vec = np.array([obs / 100])
        
        return vec

    def render(self, mode="human"):
        pass