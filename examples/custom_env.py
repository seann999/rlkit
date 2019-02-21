from garage.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
import numpy as np

def create_swingup(force=0.5):
    env = CartpoleSwingupEnv()
    
    def new_step(fn):
        def step(action):
            return fn(action * force)
        
        return step
    
    def log(path, **kwargs):
        #env.log_diagnostics(path)
        pass
    
    def is_current_done():
        return False
    
    def compute_reward(action):
        yield
        
        if np.cos(env.pole.angle) > 0.95 and abs(env.cart.position[0]) < 3 and abs(env.cart.linearVelocity[0]) < 1 and abs(env.pole.angularVelocity) < 1:
            yield 1
        else:
            yield -0.1 * np.linalg.norm(action)
    
    env.log_diagnostics = log
    env.compute_reward = compute_reward
    env.is_current_done = is_current_done
    env.step = new_step(env.step)
    
    return env