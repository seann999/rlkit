import numpy as np
import pygame

from box2d.serializable import Serializable
from box2d.box2d_env import Box2DEnv
from box2d.parser import find_body


# Tornio, Matti, and Tapani Raiko. "Variational Bayesian approach for
# nonlinear identification and control." Proc. of the IFAC Workshop on
# Nonlinear Model Predictive Control for Fast Systems, NMPC FS06. 2006.
class CartpoleSwingupSparseEnv(Box2DEnv, Serializable):
    def __init__(self, *args, **kwargs):
        super().__init__(self.model_path("cartpole.xml.mako"), *args, **kwargs)

        self.max_cart_pos = 3
        self.max_reward_cart_pos = 3
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-1, -2, np.pi - 1, -3],
            [1, 2, np.pi + 1, 3],
        ])
        low, high = bounds
        xpos, xvel, apos, avel = np.random.uniform(low, high)
        self.cart.position = (xpos, self.cart.position[1])
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        self.pole.angle = apos
        self.pole.angularVelocity = avel
        return self.get_current_obs()

    def compute_reward(self, action):
        yield

        if np.cos(self.pole.angle) > 0.95 and abs(self.cart.position[0]) < 3 and abs(self.cart.linearVelocity[0]) < 1 and abs(self.pole.angularVelocity) < 1:
            yield 1
        else:
            yield -0.01 * np.linalg.norm(action)

    def is_current_done(self):
        return False

    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-10])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+10])
        else:
            return np.asarray([0])

    def log_diagnostics(self, paths, **kwargs):
        pass
