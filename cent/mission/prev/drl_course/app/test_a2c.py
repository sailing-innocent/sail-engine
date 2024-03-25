from __future__ import annotations

# the policy gradient test case 
import pytest 

import gymnasium as gym
import torch
import numpy as np

import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from utils.algorithm.rl.actor_critic import ActorCritic

from scene.grid.point_nav_2d import PointNav2D

from itertools import count

@pytest.mark.app
def test_cartpole_a2c_train():
    env = gym.make("CartPole-v1", render_mode=None)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    running_reward = 10
    # run infinitely many episodes
    total_num_episodes = 5e3

    obs_space_dim = env.observation_space.shape[0]
    assert obs_space_dim == 4 # x, dot x and theta, dot theta 
    # action space dimension (1)
    act_space_dim = env.action_space.n
    assert act_space_dim == 2 # 0 or 1
    rewards_over_seeds = []
    # for each episode, only run 9999 steps
    seeds = [42]

    reward_threshold = env.spec.reward_threshold

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        state = env.reset()
        agent = ActorCritic(obs_space_dim, act_space_dim, cuda=False)
        reward_over_episodes = []

        print(f"training in seed {seed} in total {len(seeds)} seeds")

        for episode in tqdm(range(int(total_num_episodes))):
            obs, info = wrapped_env.reset(seed=seed)
            done = False 
            ep_reward = 0
            # collect trial-and-fault
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)
                ep_reward += reward
                done = terminated or truncated  

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update the policy network
            agent.update()

            if episode % 100 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print()
                print("Episodes:", episode, "Average Reward: ", avg_reward)

            rewards_over_seeds.append(reward_over_episodes)

            # early break if the task is solved
            if running_reward > reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="ACTOR_CRITIC for CartPole-v1"
    )
    # plt.show()
    plt.savefig("output/test_cartpole_v1.png")

    assert True 

