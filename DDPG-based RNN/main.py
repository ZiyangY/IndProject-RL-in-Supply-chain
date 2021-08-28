# -*- coding: utf-8 -*-

import os
import numpy as np
import or_gym
import gym
import torch
import DDPG, memory,PPO

import matplotlib.pyplot as plt
from matplotlib import gridspec

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10,env_config={}):
    policy.eval_mode()
    avg_reward = 0.
    env = or_gym.make(env_name,env_config=env_config)
    env.seed(seed + 100)

    for _ in range(eval_episodes):
        state, done = env.reset(), False
        hidden = None
        while not done:
            action, hidden = policy.select_action(np.array(state), hidden)
            # env.render(mode='human', close=False)
            state, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    policy.train_mode()
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def plot(rewards):

    # data.save()
    p = 20
    mean_rewards = np.array([np.mean(rewards[i-p:i+1]) 
                    if i >= p else np.mean(rewards[:i+1]) 
                    for i, _ in enumerate(rewards)])
    std_rewards = np.array([np.std(rewards[i-p:i+1])
                   if i >= p else np.std(rewards[:i+1])
                   for i, _ in enumerate(rewards)])
     
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax0 = fig.add_subplot(gs[:, :-2])
    ax0.fill_between(np.arange(len(mean_rewards)), 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     label='Standard Deviation', alpha=0.3)
    ax0.plot(mean_rewards, label='Mean Rewards')
    # ax0.plot(reward_lst,label='reward for each ep')
    ax0.set_ylabel('Rewards',fontsize=20)
    ax0.set_xlabel('Episode',fontsize=20)
    ax0.set_title('LSTM + DDPG (5 stages)', fontsize = 17)
    ax0.text
    ax0.legend(loc = 4)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.ylim([-300, 500])
    plt.grid()
    
def main():
    reward_lst = []

    environment = "InvManagement-v1"
    SEED = 0
    start_timesteps = 600
    max_timesteps = 3000
    expl_noise = 0.1
    batch_size = 1000
    memory_size = 1e6

    HIDDEN_SIZE = 256
    SAVE_MODEL = True
    load_model = ""
    #load_model = "DDPG_InvManagement-v1_0"

    file_name = f"DDPG_{environment}_{SEED}"
    print("---------------------------------------")
    print(f"Policy: DDPG, Env: {environment}, Seed: {SEED}")
    print("---------------------------------------")
    
    f=open("GRU easy problem .txt","w")
    
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if SAVE_MODEL and not os.path.exists("./models"):
        os.makedirs("./models")

        
    # env_config = {'dist':6,
    #           'dist_param' : {'scale':1,'loc':20}}

    # env_config = {'periods': 50,
    #               'I0':[100,100,100,200],
    #               'p':2,
    #               'r':[1.75, 1.5, 1.0, 0.75, 0.5],
    #               'k':[0.125, 0.10, 0.075, 0.05, 0.025],
    #               'h':[0.20, 0.15, 0.10, 0.05],
    #               'c':[110, 100, 90, 80],
    #               'L': [3, 5, 7, 10],  # max leading time doesn't change
    #               'max_rewards':1000}

    # env_config = {'periods': 10,
    #           'I0':[100,200],
    #           'r':[1.00,0.75,0.50],
    #           'k':[0.075,0.05, 0.025],
    #           'h':[0.10,0.05],
    #           'c':[90,80],
    #           'L':[3,5],
    #           'max_rewards':1000}
    env_config = {}
    env = or_gym.make(environment,env_config = env_config)
    #env = or_gym.make(environment)

    # Set seeds
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])
    

    configs = {
        #state_dim  = state_dim for default state vector size, 3 for reshape, 93 for pipeline inventory
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": 256,
        "discount": 0.99,
        "tau": 0.00001,
        "recurrent_actor":False,
        "recurrent_critic": False,
        "actor_lr":4e-5,  #policy
        "critic_lr":6e-5,   #q value
    }

    policy = DDPG.DDPG(**configs)
    
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    if load_model != "":
        policy_file = file_name \
            if load_model == "default" else load_model
        policy.load(f"./models/{file_name}")

    # if test:
    #     eval_policy(policy, environment, SEED, eval_episodes=10, test=True)
    #     return
    
    #state_dim (1st parameter  = state_dim for default state vector size, 3 for reshape, 93 for pipeline inventory
    replay_buffer = memory.ReplayBuffer(
        state_dim, action_dim, HIDDEN_SIZE,
        memory_size, recurrent=False)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, environment, SEED,env_config=env_config)]
    best_reward = evaluations[-1]
    
    
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    hidden = policy.get_initial_states()

    for t in range(1, int(max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
            _, next_hidden = policy.select_action(np.array(state), hidden)
        else:
            # a, next_hidden = policy.select_action(np.array(state), hidden)
            # action = (
            #     a + np.random.normal(
            #         0, max_action * expl_noise, size=action_dim)
            # ).clip(-max_action, max_action)
            
            #without using exploration noise case:
            action, next_hidden = policy.select_action(np.array(state), hidden)

        # Perform action
        next_state, reward, done, _ = env.step(action)

        done_bool = float(
            done) if episode_timesteps < env.periods else 0
        # Store data in replay buffer
        replay_buffer.add(
            state, action, next_state, reward, done_bool, hidden, next_hidden)

        state = next_state
        hidden = next_hidden
        episode_reward += reward
        critic_loss = 0
        actor_loss = 0
        # Train agent after collecting sufficient data
        if (not policy.on_policy) and t >= start_timesteps:
            critic_loss, actor_loss = policy.train(replay_buffer, batch_size)


        if done:
            # writer.add_scalar('reward', episode_reward, t)
            # writer.add_scalar('actor loss', actor_loss, t)
            # writer.add_scalar('critic loss', critic_loss, t)
            # +1 to account for 0 indexing. +0 on ep_timesteps since it
            #  will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            reward_lst.append(episode_reward)
            f.write(str(episode_reward))
            f.write("\n")
            #decay expl_noise during the running
            expl_noise *= 0.9
            
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            hidden = policy.get_initial_states()

        # Evaluate episode
        if (t + 1) % 3000 == 0:
            evaluations.append(eval_policy(policy,environment, SEED,env_config=env_config))
            if evaluations[-1] > best_reward and SAVE_MODEL:
                policy.save(f"./models/{file_name}")

            np.save(f"./results/{file_name}", evaluations)

    plot(reward_lst)
    print(expl_noise)
    

if __name__ == "__main__":
    main()
