import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "glfw" # for mujoco rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from pg import PG
from ddpg import DDPG
from common import helper as h
from common import logger as logger

import make_env

import time
import numpy as np

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episode=10, verbose=True):
    total_test_reward = 0
    rewards = []
    for ep in range(num_episode):
        obs, done= env.reset(), False
        test_reward = 0

        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))
            
            test_reward += reward

        total_test_reward += test_reward
        rewards.append(test_reward)
        # if verbose:
        #     print("Test ep_reward:", test_reward)

    if verbose:
        print("Average test reward:", np.mean(rewards))
        print("Test reward stddev:" , np.std(np.asarray(rewards)))

    return total_test_reward / num_episode

def main():
    agent_seeds = [1, 48, 128] # do NOT change these values
    test_seeds = [10, 20, 30] # you may change this
    print("=======================================")
    print("===      TESTING LANDER MEDIUM      ===")
    print("=======================================")
    agents = ["pg", "ddpg_400"]
    #select working dir
    work_dir = Path().cwd()/'results'/'lunarlander_continuous_medium'

    #load env
    env = make_env.create_env("lunarlander_continuous_medium", seed=1)
    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    for agent_name in agents:
        print("---------------------------------------")
        print("Testing the " + agent_name + " agent....")
        print("---------------------------------------")
        # load agent
        if agent_name == "pg":
            agent = PG(state_shape[0], action_dim, 0.003, 0.99)
        else:
            agent = DDPG(state_shape, action_dim, max_action,
                            lr=0.001, gamma=0.95, tau=0.01, batch_size=256, buffer_size=1e6,
                            actor_hidden_dim=400, critic_hidden_dim=400)
        for seed in agent_seeds:
            print("-- Loading agent trained with seed=" + str(seed) + " --")
            model_path = work_dir/'model'/f'{agent_name}_lunarlander_continuous_medium_{seed}_params.pt'
            agent.load(model_path)
            for seed in test_seeds:
                env.reset(seed=seed)
                h.set_seed(seed)
                print('Testing (seed='+ str(seed) + ') ...')
                test(agent, env, num_episode=50)

    print("=======================================")
    print("===       TESTING WALKER EASY       ===")
    print("=======================================")
    agents = ["pg", "ddpg_400_v2"]
    #select working dir
    work_dir = Path().cwd()/'results'/'bipedalwalker_easy'

    #load env
    env = make_env.create_env("bipedalwalker_easy", seed=1)
    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    for agent_name in agents:
        print("---------------------------------------")
        print("Testing the " + agent_name + " agent....")
        print("---------------------------------------")
        # load agent
        if agent_name == "pg":
            agent = PG(state_shape[0], action_dim, 0.0003, 0.99)
        else:
            agent = DDPG(state_shape, action_dim, max_action,
                            lr=5e-5, gamma=0.99, tau=0.005, batch_size=256, buffer_size=1e6,
                            actor_hidden_dim=400, critic_hidden_dim=400)
        for seed in agent_seeds:
            print("-- Loading agent trained with seed=" + str(seed) + " --")
            model_path = work_dir/'model'/f'{agent_name}_bipedalwalker_easy_{seed}_params.pt'
            agent.load(model_path)
            for seed in test_seeds:
                env.reset(seed=seed)
                h.set_seed(seed)
                print('Testing (seed='+ str(seed) + ') ...')
                test(agent, env, num_episode=50)


# Entry point of the script
if __name__ == "__main__":
    main()