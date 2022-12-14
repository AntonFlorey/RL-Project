import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl" #"glfw" # for mujoco rendering
import time
from pathlib import Path
import json

import torch
import gym
import hydra
import wandb
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ppo import PPO
from common import helper as h
from common import logger as logger
import make_env

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

# Policy training function
def train(agent: PPO, env, max_episode_steps=512, runs_per_episode=32, min_steps=1000):
    collected_reward_sums = []
    collected_timesteps = []
    run = 0
    steps = 0
    while (run < runs_per_episode or steps < min_steps):
        # Run actual training        
        reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
        # Reset the environment and observe the initial state
        obs = env.reset()
        while not done:
            episode_timesteps += 1
            steps += 1
            
            # Sample action from policy
            action, act_logprob = agent.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _ = env.step(to_numpy(action))

            if episode_timesteps >= max_episode_steps:
                done = True

            # Store action's outcome (so that the agent can improve its policy)
            agent.record(obs, action, act_logprob, reward, done, next_obs, run)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

            # update observation
            obs = next_obs.copy()
        run += 1
        collected_reward_sums.append(reward_sum)
        collected_timesteps.append(timesteps)

    # update the policy after runs have been collected
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': np.sum(collected_timesteps),
                'ep_reward': np.mean(collected_reward_sums),})
    return info


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episode=10, verbose=True):
    total_test_reward = 0
    rewards = []
    for ep in range(num_episode):
        obs, done= env.reset(), False
        test_reward = 0
        timesteps = 0
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))
            
            test_reward += reward
            timesteps += 1

        total_test_reward += test_reward
        rewards.append(test_reward)
        # if verbose:
        #     print("Test ep_reward:", test_reward)

    if verbose:
        print("Average test reward:", total_test_reward/num_episode)
        print("Test reward stddev:" , np.std(np.asarray(rewards)))

    return total_test_reward / num_episode, timesteps


# The main function
@hydra.main(config_path='cfg', config_name='project_part2')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'
    if cfg.save_model: 
        h.make_dir(work_dir/"model")
    if cfg.save_logging or cfg.save_stats: 
        h.make_dir(work_dir/"logging")
        if cfg.save_logging:
            L = logger.Logger() # create a simple logger to record stats

    if cfg.save_video:
        h.make_dir(work_dir/"video"/"train")
        h.make_dir(work_dir/"video"/"test")

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'

   # create a env
    env = make_env.create_env(cfg.env_name, seed=cfg.seed)
    if cfg.save_video and not cfg.testing:
        ep_trigger = 100
        video_path = work_dir/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=f'{cfg.agent}')

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    if not cfg.testing: # training
        for seed in cfg.seeds:
            # use wandb to store stats; we aren't currently logging anything into wandb during testing
            if cfg.use_wandb:
                wandb.init(project="rl_aalto",
                            name=f'{cfg.env_name}-{cfg.agent}-{str(seed)}-{str(cfg.run_id)}',
                            group=f'{cfg.exp_name}-{cfg.env_name}',
                            config=cfg)

            # set the current seed
            h.set_seed(cfg.seed)
            env.reset(seed=seed)

            # init agent
            agent = PPO(state_shape[0], action_dim, cfg.lr, cfg.gamma, cfg.actor_hd, cfg.critic_hd, 
                        cfg.clip_eps, cfg.gae_lambda, cfg.entropy_weight, cfg.num_minibathes, cfg.minibatch_size, cfg.grad_clipping)
            # collect some trainig data
            stats = {
                'num_episodes': cfg.train_episodes,
                'early_stop_episode': -1,
                'first_episode_over_target': -1,
                'time_for_all_episodes': 0,
                'time_until_early_stop': -1,
                'time_until_over_target': -1,
                'early_stop_avg_rew': 0,
                'end_avg_rew': 0
            }

            train_time = 0
            early_stop_count = 0
            best_avg_ret = 0
            early_stop_ep = 0
            avg_rew = 0
            running_ret = 0
            new_ret_w = 0.05
            summed_timesteps = 0

            start_time = time.thread_time()

            for ep in range(cfg.train_episodes + 1):
                # set policy exploration...
                progression = ep / cfg.train_episodes
                progression = np.clip(avg_rew / cfg.target_rew, 0, 1)
                progression = np.clip((running_ret - cfg.min_rew) / cfg.target_rew, 0, 1)
                std = (1.0 - progression) * cfg.start_exploration_std + progression * cfg.end_exploration_std
                eps = (1.0 - progression) * cfg.clip_eps + progression * cfg.end_clip_eps
                lr = (1.0 - progression) * cfg.lr + progression * cfg.end_lr
                entropy_w = (1.0 - progression) * cfg.entropy_weight
                # print("setting std to", std)
                agent.set_policy_std(std)
                agent.set_clip_eps(eps)
                agent.set_lr(lr)
                agent.entropy_weight = entropy_w

                # collect data and update the policy
                start = time.thread_time()
                train_info = train(agent, env, cfg.horizon, cfg.num_agents, cfg.min_steps_per_ep)
                end = time.thread_time()
                train_time += (end - start)
                summed_timesteps += train_info['timesteps']

                # one test run
                curr_test_rew, test_timesteps = test(agent, env, num_episode=1, verbose=False)

                # running reward 
                running_ret = new_ret_w * curr_test_rew + (1.0 - new_ret_w) * running_ret

                #store train info
                train_info['ep_reward'] = curr_test_rew
                train_info['running_reward'] = running_ret
                train_info['timesteps'] = test_timesteps
                train_info['total_time_elapsed'] = time.thread_time() - start_time

                if (curr_test_rew >= cfg.target_rew and stats['first_episode_over_target'] == -1):
                    stats['first_episode_over_target'] = ep
                    stats['time_until_over_target'] = train_time

                # early stopping
                if running_ret >= cfg.target_rew:
                    if cfg.save_model:
                        cfg.model_path = work_dir/'model'/f'{cfg.agent}_{cfg.env_name}_{seed}_params.pt'
                        agent.save(cfg.model_path)
                    stats['early_stop_avg_rew'] = test(agent, env, num_episode=50, verbose=False)[0]
                    stats['early_stop_episode'] = ep
                    stats['time_until_early_stop'] = train_time
                    break

                if cfg.use_wandb:
                    train_info['total_timesteps'] = summed_timesteps
                    wandb.log(train_info)
                if cfg.save_logging:
                    L.log(**train_info)
                if (not cfg.silent) and (ep % 20 == 0):
                    print({"ep": ep, **train_info, "std": std, "eps": eps, "lr": lr, "entropy_w": entropy_w})

            if (cfg.save_model and stats['time_until_early_stop'] == -1):
                cfg.model_path = work_dir/'model'/f'{cfg.agent}_{cfg.env_name}_{seed}_params.pt'
                agent.save(cfg.model_path)

            stats['time_for_all_episodes'] = train_time
            stats['end_avg_rew'] = test(agent, env, num_episode=50, verbose=False)[0]

            if cfg.save_stats:
                stats_path = work_dir/'logging'/f'{cfg.agent}_{cfg.env_name}_{seed}_stats.json'
                with open(stats_path, 'w') as f:
                    f.write(json.dumps(stats))

            if cfg.use_wandb:
                wandb.finish()

    else: # testing
        for seed in cfg.seeds:
            # init agent
            agent = PPO(state_shape[0], action_dim, cfg.lr, cfg.gamma, cfg.actor_hd, cfg.critic_hd, 
                        cfg.clip_eps, cfg.gae_lambda, cfg.entropy_weight, cfg.num_minibathes, cfg.minibatch_size, cfg.grad_clipping)
                               
            cfg.model_path = work_dir/'model'/f'{cfg.agent}_{cfg.env_name}_{seed}_params.pt'
            print("Loading model from", cfg.model_path, "...")
            # load model
            agent.load(cfg.model_path)
            for seed in cfg.seeds:
                env.reset(seed=seed)
                h.set_seed(seed)
                print('Testing (seed='+ str(seed) + ') ...')
                test(agent, env, num_episode=50)

# gamma = 0.99
# lam = 0.95

# def gae(rewards, values, next_values, dones):
#     deltas = [(r + (1.0 - d) * gamma * nv - v) for r, nv, v, d in zip(rewards, values, next_values, dones)]
#     print("deltas: ", deltas)

#     gaes = deltas.copy()
#     for t in reversed(range(len(deltas)-1)):
#         print(t)
#         gaes[t] = gaes[t] + (1 - dones[t]) * (lam * gamma * gaes[t+1])
#     print("gaes: ", gaes)

# Entry point of the script
if __name__ == "__main__":
    # rew = [1,1,1,1]
    # values = [0,0,0,0]
    # next_values = [0,0,0,0]
    # dones = [0,0,0,1]
    # gae(rew, values, next_values, dones)
    main()


