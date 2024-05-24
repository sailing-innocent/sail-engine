from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
import pytest 

from tqdm import tqdm


plt.rcParams["figure.figsize"] = (10, 5)

# class PolicyNetwork

class PolicyNetwork(nn.Module):
    """Parameterize Policy Network. """
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
        of a normal distribution from which an action is sample from.
        """

        super().__init__()
        hidden_space1 = 16 
        hidden_space2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh()
        )

        # Policy Mean Specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Stardard Derivitive Liear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned On the observation, returns the mean and standard derivative
        of a normal distribution from which an action is sampled

        Args:
            x: Observation from the environment
        
        Returns:
            action_mean: predicted mean of the normal distribution
            action_stddevs: predict the standard derivation of the normal distribution
        """

        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm"""

    def __init__(self, obs_space_dims: int, action_space_dims: int, cuda: bool = False):
        """Initialize the agent that learns a policy using REINFORCE algorithm
        """

        # Hyperparameters
        self.lr = 1e-4 # learning rate
        self.gamma = 0.99
        self.eps = 1e-6
        self.cuda = cuda

        self.probs = []
        self.rewards = []
        self.net = PolicyNetwork(obs_space_dims, action_space_dims)
        if cuda:
            self.net = self.net.cuda()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation

        Args:
            state: Observation from the environment
        
        Returns:
            action: Actions to be performed
        """

        state = torch.Tensor(np.array([state]))
        if self.cuda:
            state = state.cuda()
        
        action_means, action_stddevs = self.net(state)

        if self.cuda:
            action_means = action_means.cpu()
            action_stddevs = action_stddevs.cpu()

        assert action_means.shape == torch.Size([1,1])
        assert action_stddevs.shape == torch.Size([1,1])

        # create a normal distribution from predicted mean and standard derivation

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Update the PolicyNetwork weights"""

        running_g = 0
        gs = []

        # Discounted return (backwards) [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty the state
        self.probs = []
        self.rewards = []

@pytest.mark.app
def test_inverted_pendium():
    env = gym.make("InvertedPendulum-v4", render_mode=None)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    # total number of episodes to train
    # total_num_episodes = int(5e3)
    total_num_episodes = int(5e1)

    # observation space dimension (4)
    obs_space_dims = env.observation_space.shape[0]
    assert obs_space_dims == 4
    # action space dimension (1)
    act_space_dims = env.action_space.shape[0]
    assert act_space_dims == 1

    rewards_over_seeds = []
    for seed in [1, 2]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # reinit agent for every seed 
        agent = REINFORCE(obs_space_dims, act_space_dims)
        reward_over_episodes = []

        print(f"training in seed {seed} in total {2} seeds")

        for episode in tqdm(range(total_num_episodes)):
            obs, info = wrapped_env.reset(seed=seed)

            done = False 
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)
                done = terminated or truncated 


            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update the policy network
            agent.update()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episodes:", episode, "Average Reward: ", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for InvertedPendulum-v4"
    )
    plt.show()

    assert True 

@pytest.mark.current
def test_inverted_pendium_cuda():
    env = gym.make("InvertedPendulum-v4", render_mode=None)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    # total number of episodes to train
    # total_num_episodes = int(5e3)
    total_num_episodes = int(5e1)

    # observation space dimension (4)
    obs_space_dims = env.observation_space.shape[0]
    assert obs_space_dims == 4
    # action space dimension (1)
    act_space_dims = env.action_space.shape[0]
    assert act_space_dims == 1

    rewards_over_seeds = []
    for seed in [1, 2]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # reinit agent for every seed 
        agent = REINFORCE(obs_space_dims, act_space_dims, cuda=True)
        reward_over_episodes = []

        print(f"training in seed {seed} in total {2} seeds")

        for episode in tqdm(range(total_num_episodes)):
            obs, info = wrapped_env.reset(seed=seed)

            done = False 
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)
                done = terminated or truncated 

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update the policy network
            agent.update()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episodes:", episode, "Average Reward: ", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for InvertedPendulum-v4"
    )
    # plt.show()
    plt.savefig("test_inverted_pendium_cuda.png")

    assert True 
