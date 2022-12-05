import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl" # for mujoco rendering
import time
from pathlib import Path
import json

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ppo import PPO
from common import helper as h
from common import logger as logger
import make_env

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

# Policy training function
def train(agent: PPO, env, max_episode_steps=1000):
    # Run actual training        
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
    while not done:
        episode_timesteps += 1
        
        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

        # Store action's outcome (so that the agent can improve its policy)
        agent.record(obs, action, act_logprob, reward, done, next_obs)

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

    # update the policy after one episode TODO: maybe after multiple parallel episodes as proposed in the paper
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episode=10):
    total_test_reward = 0
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
        print("Test ep_reward:", test_reward)

    print("Average test reward:", total_test_reward/num_episode)


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

    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)

   # create a env
    env = make_env.create_env(cfg.env_name, seed=cfg.seed)


    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name) # save video every 50 episode

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
            agent = PPO(state_shape, action_dim, cfg.lr, cfg.gamma, cfg.actor_hd, cfg.critic_hd, 
                        cfg.clip_eps, cfg.entropy_weight, cfg.num_minibathes, cfg.minibatch_size)
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

            for ep in range(cfg.train_episodes + 1):
                # collect data and update the policy
                start = time.thread_time()
                train_info = train(agent, env, ep)
                end = time.thread_time()
                train_time += (end - start)

                if cfg.use_wandb:
                    wandb.log(train_info)
                if cfg.save_logging:
                    L.log(**train_info)
                if (not cfg.silent) and (ep % 100 == 0):
                    print({"ep": ep, **train_info})

                if (train_info['ep_reward'] >= cfg.target_rew and stats['first_episode_over_target'] == -1):
                    stats['first_episode_over_target'] = ep
                    stats['time_until_over_target'] = train_time

                if (ep % cfg.test_interval == 0):
                    avg_rew = test(agent, env, num_episode=100, verbose=False)
                    print("Test performance is " + str(avg_rew))
                    if (avg_rew > best_avg_ret):
                        best_avg_ret = avg_rew
                        early_stop_ep = ep
                        early_stop_count = 0
                        # save this state of the model
                        if (best_avg_ret >= cfg.target_rew):
                            if cfg.save_model:
                                cfg.model_path = work_dir/'model'/f'{cfg.agent}_{cfg.env_name}_{seed}_params.pt'
                                agent.save(cfg.model_path)
                            stats['early_stop_avg_rew'] = avg_rew
                            stats['early_stop_episode'] = early_stop_ep
                            stats['time_until_early_stop'] = train_time
                            # lower learning rate to focus on improving the found solution
                            #agent.adjust_learning_rate(0.1)
                    else:
                        early_stop_count += cfg.test_interval
                        # early stopping
                        if (early_stop_count >= cfg.early_stop_episodes and best_avg_ret >= cfg.target_rew):
                            print("Stopped early in episode: " + str(early_stop_ep))
                            break

            if (cfg.save_model and best_avg_ret < cfg.target_rew):
                cfg.model_path = work_dir/'model'/f'{cfg.agent}_{cfg.env_name}_{seed}_params.pt'
                agent.save(cfg.model_path)

            stats['time_for_all_episodes'] = train_time
            stats['end_avg_rew'] = test(agent, env, num_episode=10, verbose=False)

            if cfg.save_stats:
                stats_path = work_dir/'logging'/f'{cfg.agent}_{cfg.env_name}_{seed}_stats.json'
                with open(stats_path, 'w') as f:
                    f.write(json.dumps(stats))

            if cfg.use_wandb:
                wandb.finish()

    else: # testing
        for seed in cfg.seeds:
            # init agent
            agent = PPO(state_shape, action_dim, cfg.lr, cfg.gamma, cfg.actor_hd, cfg.critic_hd, 
                        cfg.clip_eps, cfg.entropy_weight, cfg.num_minibathes, cfg.minibatch_size)
                               
            cfg.model_path = work_dir/'model'/f'{cfg.agent}_{cfg.env_name}_{seed}_params.pt'
            print("Loading model from", cfg.model_path, "...")
            # load model
            agent.load(cfg.model_path)
            for seed in cfg.seeds:
                env.reset(seed=seed)
                h.set_seed(seed)
                print('Testing (seed='+ str(seed) + ') ...')
                test(agent, env, num_episode=50)


# Entry point of the script
if __name__ == "__main__":
    main()


