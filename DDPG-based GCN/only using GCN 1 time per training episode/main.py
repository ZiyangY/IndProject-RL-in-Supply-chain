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

# p = [positive float] unit price for final product.
# r = [non-negative float; dimension |Stages|] unit cost for replenishment orders at each stage.
R = [0.75,0.5]
# k = [non-negative float; dimension |Stages|] backlog cost or goodwill loss (per unit) for unfulfilled orders (demand or replenishment orders).
# h = [non-negative float; dimension |Stages|-1] unit holding cost for excess on-hand inventory at each stage.
#     (Note: does not include pipeline inventory).
H = [0.05]
# c = [positive integer; dimension |Stages|-1] production capacities for each suppliers (stages 1 through |Stage|).
C = [80]
# L = [non-negative integer; dimension |Stages|-1] lead times in betwen stages.
L = [10]


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
                
            inventory=np.append(inventory,2000)
            replenishment = np.append(replenishment, 0)
            last_replenishment = np.append(last_replenishment, 0)
            
            
            
            
            
            action, s_g = policy.select_action(inventory,replenishment,last_replenishment,unites_sold,lost_sales)

            
            # action, hidden = policy.select_action(np.array(state), hidden)
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
    ten_1 = torch.full([1,len(inventory)*2 - 2],0)
    ten_2 = torch.full([1,len(inventory)*2 - 2],0)
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
    
    
    a = torch.full([len(inventory),5],0.0)

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
    # ax0.set_title('GCN + DDPG', fontsize = 17)
    ax0.text
    ax0.legend(loc = 4)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.grid()
     
                                   


def main():
    output = open('optimaze.txt','w',encoding='gbk')
    reward_lst = []

    environment = "InvManagement-v1"
    SEED = 30
    start_timesteps = 1000
    max_timesteps = 8000
    expl_noise = 0.2
    batch_size = 3000
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
    #           'L':[5,10],
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
        # "actor_lr":4e-7,  #policy  -5能学
        # "critic_lr":6e-7,   #q value
        "actor_lr":2e-6,  #policy
        "critic_lr":5e-6,   #q value
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
        3, action_dim, HIDDEN_SIZE,
        memory_size, recurrent=False)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, environment, SEED,env_config=env_config)]
    best_reward = evaluations[-1]
    
    
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    # hidden = policy.get_initial_states()

        
        
    for t in range(1, int(max_timesteps)):
        episode_timesteps += 1
        s_g = None
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
            # _, next_hidden = policy.select_action(np.array(state))
        else:
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
                
            inventory=np.append(inventory,2000)
            replenishment = np.append(replenishment, 0)
            last_replenishment = np.append(last_replenishment, 0)
            
            
            
            
            
            a, s_g = policy.select_action(inventory,replenishment,last_replenishment,unites_sold,lost_sales)
            action = (
                a + np.random.normal(
                    0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            
            #without using exploration noise case:
            #action, next_hidden = policy.select_action(np.array(state), hidden)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        p = env.period-1
        if (p!=0):
            s_g_1 = create_torch_graph_data(next_state,env.R[p],env.R[p-1],env.S[p],env.LS[p])
        else:
            s_g_1 = create_torch_graph_data(next_state,env.R[p],
                                            env.R[0],env.S[p],env.LS[p])
        
        
        done_bool = float(
            done) if episode_timesteps < env.periods else 0
        # Store data in replay buffer
        replay_buffer.add(
            state, s_g, action, next_state,s_g_1, reward, done_bool)

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
        if (t + 1) % 3000 == 0:
            evaluations.append(eval_policy(policy,environment, SEED,env_config=env_config))
            if evaluations[-1] > best_reward and SAVE_MODEL:
                policy.save(f"./models/{file_name}")

            np.save(f"./results/{file_name}", evaluations)

    plot(reward_lst)
    print(expl_noise)
    

if __name__ == "__main__":
    main()
