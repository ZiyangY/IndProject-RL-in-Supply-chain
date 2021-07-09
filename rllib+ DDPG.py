# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:17:48 2021

@author: Laura Y
"""
import or_gym
from or_gym.utils import create_env
import ray
from ray.rllib import agents
from ray import tune
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

env_name = 'InvManagement-v1'
# env_config = {'dist':2,
#               'dist_param' : {'n': 20,'p':0.9}}
# env_config = {'dist':6,
#               'dist_param' : {'scale':1,'loc':20}}
env_config = {'periods': 10,
              'I0':[100,200],
              'r':[1.00,0.75,0.50],
              'k':[0.075,0.05, 0.025],
              'h':[0.10,0.05],
              'c':[90,80],
              'L':[5,10],
              'max_rewards':1000}
def register_env(env_name, env_config=env_config):
    env = create_env(env_name)
    tune.register_env(env_name,
        lambda env_name: env(env_name,
            env_config=env_config))

rl_config = dict(
    env=env_name,
    num_workers=8,
    env_config=env_config,
    actor_lr= 1e-4,
    critic_lr= 1e-4,
    gamma= 0.99,
    buffer_size=10000,
    learning_starts=700,
    tau= 0.001,
    train_batch_size=64,
    actor_hiddens=[400,300],
    critic_hiddens=[400,300],
)
# Register environment
register_env(env_name, env_config)

ray.init(ignore_reinit_error=True)
agent = agents.ddpg.DDPGTrainer(env=env_name, config=rl_config)
results = []
for i in range(600):
    res = agent.train()
    results.append(res)
    print('\rIter: {}\tReward: {:.2f}'.format(i + 1, res['episode_reward_mean']), end='')
    
# Unpack values from each iteration
rewards = np.hstack([i['hist_stats']['episode_reward'] 
    for i in results])
# pol_loss = [
#     i['info']['learner']['default_policy']['policy_loss'] 
#     for i in results]
# vf_loss = [
#     i['info']['learner']['default_policy']['vf_loss'] 
#     for i in results]
 
p = 100
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
ax0.set_ylabel('Rewards')
ax0.set_xlabel('Episode')
ax0.set_title('Training Rewards')
ax0.legend()
 
# ax1 = fig.add_subplot(gs[0, 2:])
# ax1.plot(pol_loss)
# ax1.set_ylabel('Loss')
# ax1.set_xlabel('Iteration')
# ax1.set_title('Policy Loss')
 
# ax2 = fig.add_subplot(gs[1, 2:])
# ax2.plot(vf_loss)
# ax2.set_ylabel('Loss')
# ax2.set_xlabel('Iteration')
# ax2.set_title('Value Function Loss')
 
plt.show()