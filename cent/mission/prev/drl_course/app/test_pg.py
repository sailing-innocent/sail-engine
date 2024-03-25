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

from utils.algorithm.rl.policy_gradient import REINFORCE

from scene.grid.point_nav_2d import PointNav2D

@pytest.mark.app
def test_mujoco_invp_pg_cuda():
    env = gym.make("InvertedPendulum-v4", render_mode=None)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    # total number of episodes to train
    # total_num_episodes = int(5e3)
    total_num_episodes = int(5e1)

    # observation space dimension (4)
    obs_space_dim = env.observation_space.shape[0]
    assert obs_space_dim == 4
    # action space dimension (1)
    act_space_dim = env.action_space.shape[0]
    assert act_space_dim == 1

    rewards_over_seeds = []
    for seed in [1, 2]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # reinit agent for every seed 
        agent = REINFORCE(obs_space_dim, act_space_dim, cuda=True)
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

@pytest.mark.current
def test_mujoco_hc_pg_cuda_train():
    # https://gymnasium.farama.org/environments/mujoco/half_cheetah/
    env = gym.make("HalfCheetah-v4", render_mode="human")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 5)  # Records episode-reward

    # total number of episodes to train
    total_num_episodes = int(5e3)

    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.shape[0]

    rewards_over_seeds = []
    for seed in [42]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # reinit agent for every seed 
        agent = REINFORCE(obs_space_dim, act_space_dim, cuda=True)
        reward_over_episodes = []

        print(f"training in seed {seed} in total {1} seeds")

        for episode in tqdm(range(total_num_episodes)):
            obs, info = wrapped_env.reset(seed=seed)

            done = False 
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                # if larger than 1000 iter, whill be truncated
                agent.rewards.append(reward)
                done = terminated or truncated 

            if episode % 50 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print()
                print(f"Update the policy with {len(agent.rewards)} trajectory")
                print("Episodes:", episode, "Average Reward: ", avg_reward)

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update the policy network
            agent.update()

            if episode % 1000 == 0:
                print()
                wpath = f"output/halfcheetah_cuda_5e3_5en4_{episode}.pth"
                print(f"Saved the model at {wpath}")
                agent.save(wpath)

        rewards_over_seeds.append(reward_over_episodes)

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for HalfCheetah-v4"
    )
    # plt.show()
    plt.savefig("output/train_halfcheetah_cuda_5e3_5en4.png")

    assert True 

@pytest.mark.app
def test_load_mujoco_hc_pg_cuda():
# https://gymnasium.farama.org/environments/mujoco/half_cheetah/
    env = gym.make("HalfCheetah-v4", render_mode="human")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 5)  # Records episode-reward

    # total number of episodes to test
    total_num_episodes = int(5e1)

    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.shape[0]

    rewards_over_seeds = []
    for seed in [42]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # reinit agent for every seed 
        agent = REINFORCE(obs_space_dim, act_space_dim, cuda=True)
        agent.load("output/halfcheetah_cuda_5e3_5en4_10.pth")
        reward_over_episodes = []

        print(f"load in seed {seed} in total {1} seeds")

        for episode in tqdm(range(total_num_episodes)):
            obs, info = wrapped_env.reset(seed=seed)

            done = False 
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                # if larger than 1000 iter, whill be truncated
                agent.rewards.append(reward)
                done = terminated or truncated 

            if episode % 10 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print()
                print("Episodes:", episode, "Average Reward: ", avg_reward)

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update the policy network
        rewards_over_seeds.append(reward_over_episodes)

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for HalfCheetah-v4"
    )
    # plt.show()
    plt.savefig("output/test_halfcheetah_cuda_5e3_5en4.png")

    assert True 


@pytest.mark.app
def test_train_nav2d():
    env = PointNav2D(render_mode=None)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 5)  # Records episode-reward

    # total number of episodes to train
    total_num_episodes = int(5e2)

    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.shape[0]
    assert obs_space_dim == 4
    assert act_space_dim == 2

    rewards_over_seeds = []
    for seed in [42]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # reinit agent for every seed 
        agent = REINFORCE(obs_space_dim, act_space_dim, cuda=True)
        reward_over_episodes = []

        print(f"training in seed {seed} in total {1} seeds")

        for episode in tqdm(range(total_num_episodes)):
            obs, info = wrapped_env.reset(seed=seed)

            done = False 
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                # if larger than 1000 iter, whill be truncated
                agent.rewards.append(reward)
                done = terminated or truncated 

            if episode % 50 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print()
                print(f"Update the policy with {len(agent.rewards)} trajectory")
                print("Episodes:", episode, "Average Reward: ", avg_reward)

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update the policy network
            agent.update()

            if episode % 100 == 0:
                print()
                wpath = f"output/nav2d_cuda_5e3_5en4_{episode}.pth"
                print(f"Saved the model at {wpath}")
                agent.save(wpath)

        rewards_over_seeds.append(reward_over_episodes)

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for Custom Nav2d Maze"
    )
    # plt.show()
    plt.savefig("output/train_nav2d_cuda_5e2_5en4.png")

    assert True 

@pytest.mark.app
def test_load_nav2d_pg_cuda():
    env = PointNav2D(render_mode='human')
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 5)  # Records episode-reward

    # total number of episodes to test
    total_num_episodes = int(5e1)

    obs_space_dim = env.observation_space.shape[0]
    act_space_dim = env.action_space.shape[0]

    rewards_over_seeds = []
    for seed in [42]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # reinit agent for every seed 
        agent = REINFORCE(obs_space_dim, act_space_dim, cuda=True)
        agent.load("output/nav2d_cuda_5e3_5en4_400.pth")
        reward_over_episodes = []

        print(f"load in seed {seed} in total {1} seeds")

        for episode in tqdm(range(total_num_episodes)):
            obs, info = wrapped_env.reset(seed=seed)

            done = False 
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                # if larger than 1000 iter, whill be truncated
                agent.rewards.append(reward)
                done = terminated or truncated 

            if episode % 10 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print()
                print("Episodes:", episode, "Average Reward: ", avg_reward)

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            # update the policy network
        rewards_over_seeds.append(reward_over_episodes)

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for Nav2D"
    )
    # plt.show()
    plt.savefig("output/test_nav2d_cuda_5e3_5en4.png")

    assert True 