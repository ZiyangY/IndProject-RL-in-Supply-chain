# -*- coding: utf-8 -*-

import os
import numpy as np
import or_gym
import gym
import torch
import DDPG, memory

import matplotlib.pyplot as plt
from matplotlib import gridspec

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
# from torch_geometric.nn import ModuleList
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import grid
device= "cpu"



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

def create_torch_graph_data(inventory, replenishment,last_replenishment,unites_sold, lost_sales):    
    # edge_index that neighbour node connected
    ten_1 = torch.full([1,len(inventory)*2 - 2],0).to(device)
    ten_2 = torch.full([1,len(inventory)*2 - 2],0).to(device)
    for i in range (0,len(inventory)):
        if i ==  0:
            ten_1[0][0]=i
            ten_2[0][0]=i+1
        elif i== len(inventory)-1:
            ten_1[0][-1]=i
            ten_2[0][-1]=i-1
        else:
            ten_1[0][1+(i-1)*2]=i
            ten_2[0][1+(i-1)*2]=i-1
            
            ten_1[0][2+(i-1)*2]=i
            ten_2[0][2+(i-1)*2]=i+1            
    edge_index = torch.cat((ten_1,ten_2),0)
    
    
    a = torch.full([len(inventory),5],0.0).to(device)

    for i in range (0, len(inventory)):
        a[i] = torch.tensor([inventory[i],replenishment[i],last_replenishment[i],
                             unites_sold[i],lost_sales[i]])
    data = Data(x=a, edge_index=edge_index)

    return data            


def plot(reward_lst):
    p = 1
    mean_rewards = np.array([np.mean(reward_lst[i-p:i+1]) 
                    if i >= p else np.mean(reward_lst[:i+1]) 
                    for i, _ in enumerate(reward_lst)])
    std_rewards = np.array([np.std(reward_lst[i-p:i+1])
                   if i >= p else np.std(reward_lst[:i+1])
                   for i, _ in enumerate(reward_lst)])
    
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax0 = fig.add_subplot(gs[:, :-2])
    ax0.fill_between(np.arange(len(mean_rewards)), 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     label='Standard Deviation', alpha=0.3)
    ax0.plot(mean_rewards, label='Mean Rewards')
    ax0.set_ylabel('Rewards',fontsize=20)
    ax0.set_xlabel('Episode',fontsize=20)
    # ax0.set_title('GCN + DDPG, alr:', actor_lr, " clr:", critic_lr, fontsize = 17)
    ax0.text
    ax0.legend(loc = 4)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.grid()
     
                                   


def main():
    output = open('5 stages longer.txt','w',encoding='gbk')
    reward_lst = []

    environment = "InvManagement-v1"
    SEED = 0
    start_timesteps = 3500
    max_timesteps = 49000
    expl_noise = 0.2
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
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if SAVE_MODEL and not os.path.exists("./models"):
        os.makedirs("./models")

    # env_config = {'periods': 10,
    #           'I0':[100,200],
    #           'r':[1.00,0.75,0.50],
    #           'k':[0.075,0.05, 0.025],
    #           'h':[0.10,0.05],
    #           'c':[90,80],
    #           'L':[5,10],
    #           'max_rewards':1000}
    
    # env_config = {'periods': 70,
    #       'I0':[100,100,100,200],
    #       'r':[2.50, 2.00,1.00,0.75,0.50],
    #       'k':[0.15,0.10,0.075,0.05, 0.025],
    #       'h':[0.3,0.15,0.10,0.05],
    #       'c':[130,100,90,80],
    #       'L':[5,7,10,15],
    #       'max_rewards':1000}
    # env_config = {'periods': 70,
    #           'I0':[50,70,100,100,200],
    #           'r':[3.00,2.50, 2.00,1.00,0.75,0.50],
    #           'k':[0.3,0.15,0.10,0.075,0.05, 0.025],
    #           'h':[0.33,0.3,0.15,0.10,0.05],
    #           'c':[140,120,100,90,80],
    #           'L':[4,5,7,10,15],
    #           'max_rewards':1000}
    env_config = {}
    env = or_gym.make(environment,env_config = env_config)
    #env = or_gym.make(environment)

    # Set seeds
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = env.observation_space.shape[0]
    print("state_dim",state_dim)
    action_dim = env.action_space.shape[0]
    print("action_dim", action_dim)
    max_action = float(env.action_space.high[0])
    # actor_lr=4e-6,  #policy
    # critic_lr=6e-6,   #q value

    configs = {
        #state_dim  = state_dim for default state vector size, 3 for reshape, 93 for pipeline inventory
        "state_dim": 5,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": 256,
        "discount": 0.99,
        "tau": 0.00001,
        "recurrent_actor":False,
        "recurrent_critic": False,
        # worked version
        "actor_lr":4e-4,  #policy  -5能学 #1e-4,4e-4
        "critic_lr":8e-4,   #q value
        # "actor_lr":4e-3,  #policy
        # "critic_lr":6e-3,   #q value
        "batch_size": batch_size,
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
        6, action_dim, HIDDEN_SIZE,
        memory_size, recurrent=False)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, environment, SEED,env_config=env_config)]
    # best_reward = evaluations[-1]
    
    
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    # hidden = policy.get_initial_states()

        
        
    for t in range(1, int(max_timesteps)):
        episode_timesteps += 1
        p = env.period-1
        if (p!= 0):
            inventory = state 
            replenishment = env.R[p-1]
            unites_sold = env.S[p-1]
            lost_sales = env.LS[p-1]
        else:
            inventory = state
            replenishment = env.R[0]
            unites_sold = env.S[0]
            lost_sales = env.LS[0]
        if (p < 2):
            last_replenishment = env.R[0]
        else:
            last_replenishment = env.R[p-2]
        # lastr = [[0]*(env.lead_time.max()-1)]*4
        # if p>=env.lead_time.max():
        #     lastr = env.R[]
        # prev = p-env.lead_time.max()
        # if prev>=0:
        #     last_replenishment = env.R[prev:p-1]
        # elif prev == -11:

        #     last_replenishment = [[0]*np.abs(prev+2)]*3
        # else:
        #     last_replenishment = [[0]*np.abs(prev+2)]*3 +env.R[0:p-1]

        inventory=np.append(inventory,2000)
        replenishment = np.append(replenishment, 0)
        last_replenishment = np.append(last_replenishment, 0)
        # last_replenishment = last_replenishment+ [[0]*env.lead_time.max()]
        s_g = create_torch_graph_data(inventory,replenishment,last_replenishment,unites_sold,lost_sales)
        
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
            # _, next_hidden = policy.select_action(np.array(state))
        else:      
            # a = policy.select_action(s_g)
            # action = (
            #     a + np.random.normal(
            #         0, max_action * expl_noise, size=action_dim)
            # ).clip(-max_action, max_action)
            
            #without using exploration noise case:
            action = policy.select_action(s_g)
            
        # Perform action
        next_state, reward, done, _ = env.step(action)
        next_inventory=np.append(next_state,2000)
        p = env.period-1

        # prev = p-env.lead_time.max()
        # if prev>=0:
        #     last_replenishment = env.R[prev:p-1]
        # elif prev < -9:

        #     last_replenishment = [[0]*np.abs(prev+2)]*3
        # else:
        #     print(p)
        #     print(env.R[0:p-1])
        #     print([[0]*np.abs(prev+2)]*3)
        #     last_replenishment = [[0]*np.abs(prev+2)]*3 +env.R[0:p-1]
        # last_replenishment = last_replenishment+ [[0]*env.lead_time.max()]
        # print(last_replenishment)
        rep = np.append(env.R[p],0)
        if (p!=0):
            lastRep = np.append(env.R[p-1],0)
            s_g_1 = create_torch_graph_data(next_inventory,rep,lastRep,env.S[p],env.LS[p])
        else:
            lastRep = np.append(env.R[0],0)
            s_g_1 = create_torch_graph_data(next_inventory,rep,
                                            lastRep,env.S[p],env.LS[p])
        
        
        done_bool = float(
            done) if episode_timesteps < env.periods else 0
        # Store data in replay buffer
        replay_buffer.add(
            inventory, s_g, action, next_inventory,s_g_1, reward, done_bool)

        state = next_state
        episode_reward += reward
        critic_loss = 0
        actor_loss = 0
        # Train agent after collecting sufficient data
        if (not policy.on_policy) and t >= start_timesteps:
            critic_loss, actor_loss = policy.train(replay_buffer, batch_size)
        if done:
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            reward_lst.append(episode_reward)
            output.write(str(episode_reward))
            output.write('\n')
            #decay expl_noise during the running
            expl_noise *= 0.95
            
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            #@ hidden = policy.get_initial_states()

        # Evaluate episode
        # if (t + 1) % 3000 == 0:
        #     evaluations.append(eval_policy(policy,environment, SEED,env_config=env_config))
        #     if evaluations[-1] > best_reward and SAVE_MODEL:
        #         policy.save(f"./models/{file_name}")

        #     np.save(f"./results/{file_name}", evaluations)

    plot(reward_lst)

    print(expl_noise)
    

if __name__ == "__main__":
    main()
