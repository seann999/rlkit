class PriorQ:
    def __init__(self, q, p):
        self.q = q
        self.p = p

class TwinQ:
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

class AC:
    def __init__(self, qf, policy):
        self.qf = qf
        self.policy = policy
        self.train_policy_with_reparameterization = True
        
    def train_q(obs, actions, reward, next_obs):
        q_pred = self.qf(obs, actions)
        
        policy_outputs = self.policy(
            obs,
            reparameterize=self.train_policy_with_reparameterization,
            return_log_prob=True,
        )
        
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        
        next_q_new_actions = get_q(self.target_qf1, self.target_qf2, next_obs, next_new_actions)
    
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
            alpha = self.alpha
            alpha_loss = 0