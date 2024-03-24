import torch 
import torch.nn as nn 

from .base import AgentConfigBase, AgentBase

class PolicyNet(nn.Module):
    def __init__(self, 
                obs_dim: int, 
                act_dim: int, 
                hidden_dim_1=64, 
                hidden_dim_2=64):
        super(PolicyNet, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_1),
            nn.Tanh(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Tanh(),
        )
        self.policy_mean_net = nn.Linear(hidden_dim_2, act_dim)
        self.policy_std_net = nn.Linear(hidden_dim_2, act_dim)

    def forward(self, obs):
        feature = self.shared_net(obs)
        mean = self.policy_mean_net(feature)
        std = torch.log(1 + torch.exp(self.policy_std_net(feature)))
        return mean, std
    
class REINFORCE_AgentConfig(AgentConfigBase):
    def __init__(self, env_config, device="cuda", eps=1e-6, lr=0.01, gamma=0.99):
        super(REINFORCE_AgentConfig, self).__init__(env_config)
        self.device = device
        self.eps = eps # converge threshold
        self.lr = lr # learning rate
        self.gamma = gamma # discount factor

class REINFORCE_Agent(AgentBase):
    def __init__(self, config: REINFORCE_AgentConfig):
        super(REINFORCE_Agent, self).__init__(config)
        self.policy_net = PolicyNet(config.env_config.obs_dim, config.env_config.act_dim).to(config.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.reset()

    def reset(self):
        self.probs = []
        self.rewards = []

    def collect_reward(self, reward):
        self.rewards.append(reward)

    def observe(self, obs):
        self.state = torch.tensor(obs, dtype=torch.float32).to(self.config.device)

    def act(self):
        mean, std = self.policy_net(self.state)
        dist = torch.distributions.Normal(mean + self.config.eps, std + self.config.eps)
        action = dist.sample()
        self.probs.append(dist.log_prob(action))
        return action.cpu().numpy()

    def update(self):
        # backward network according to probs and rewards
        pass 