# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:53:38 2021
@author: Laura Y
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device= "cpu"
#device = torch.device("cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

    
class Actor(nn.Module):
    def __init__(
        self, state_dim,state_dim_2, action_dim, hidden_dim, max_action
    ):
        super(Actor, self).__init__()


        self.conv1 = GCNConv(state_dim, hidden_dim).to(device)
        self.layer_norm = LayerNorm(hidden_dim).to(device)

        self.l1 = nn.Linear(state_dim_2, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, x= None, edge_index= None,squeeze=True, bs=200, state = None,is_linear=False):
        if is_linear:
            x=F.relu(self.l1(state))
        else:
            x = F.relu(self.conv1(x,edge_index))
            if squeeze:
                x = global_add_pool(self.layer_norm(x), torch.LongTensor([0 for _ in range(len(x))]).to(device))
            else:
                x = global_add_pool(self.layer_norm(x), torch.LongTensor([bs-1 for _ in range(len(x))]).to(device))

        a = F.relu(self.l2(x))
        a = torch.tanh(self.l3(a))
        return self.max_action * a



class Critic(nn.Module):
    def __init__(
        self, state_dim, state_dim_2, action_dim, hidden_dim, is_recurrent=True
    ):
        super(Critic, self).__init__()
        self.recurrent = is_recurrent

        self.conv1 = GCNConv(state_dim, hidden_dim)
        self.l1 = nn.Linear(state_dim_2, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        self.layer_norm = LayerNorm(hidden_dim)

    def forward(self, state, action,bs,squeeze = True, is_linear=False):
        sa = torch.cat([state, action], -1)
        if is_linear:
            x = F.relu(self.l1(sa))
        else:
            dataset = []
            for i in sa:
                dataset.append(create_torch_graph_data(i))
            loader = DataLoader(dataset,bs, shuffle=False)
            for i in loader:
                x = F.relu(self.conv1(i.x,i.edge_index))
            if squeeze:
                x = global_add_pool(self.layer_norm(x), torch.LongTensor([0 for _ in range(len(x))]).to(device))
            else:
                x = global_add_pool(self.layer_norm(x), torch.LongTensor([bs-1 for _ in range(len(x))]).to(device))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim, discount,
        tau, actor_lr, critic_lr, recurrent_actor,recurrent_critic,batch_size):
        self.actor_lr= actor_lr
        self.critic_lr = critic_lr
        self.on_policy = False
        self.recurrent = recurrent_actor
        self.actor = Actor(
            state_dim, 3,action_dim, hidden_dim, max_action,
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.actor_lr)

        self.critic = Critic(
            state_dim, 3+action_dim,action_dim, hidden_dim,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.critic_lr)

        self.discount = discount
        self.tau = tau
        
            
        # s_dim = 5
        # a_dim = self.env.action_space.shape[0]

        self.batch_size = batch_size
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def get_initial_states(self):
        h_0, c_0 = None, None
        if self.actor.recurrent:
            h_0 = torch.zeros((
                self.actor.l1.num_layers,
                1,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=device)

            c_0 = torch.zeros((
                self.actor.l1.num_layers,
                1,
                self.actor.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=device)
        return (h_0, c_0)

    def select_action(self, inventory,replenishment,last_replenishment,unites_sold,lost_sales, test=True):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        s_g = create_torch_graph_data(inventory,replenishment,last_replenishment,unites_sold,lost_sales)            
        a0 = self.actor(s_g.x, s_g.edge_index).squeeze(0).detach().cpu().numpy()
        return a0,s_g

    def train(self, replay_buffer, batch_size):

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size)
        # state_dataset = []
        # for i in state:
        #     state_dataset.append(create_torch_graph_data(i))
        # state_loader = DataLoader(state_dataset,self.batch_size, shuffle=False)
        
        # next_state_dataset = []
        # for i in next_state:
        #     next_state_dataset.append(create_torch_graph_data(i))
        # next_state_loader = DataLoader(next_state_dataset,self.batch_size, shuffle=False)

        # for j in next_state_loader:
        #     next_action = torch.zeros((self.batch_size,2),dtype=torch.float).to(device)     
        #     next_action = self.actor_target(j.x,j.edge_index,squeeze=False,bs = self.batch_size).detach()

        # target_Q = reward + self.discount * self.critic_target(next_state, next_action, bs = self.batch_size).detach()
        
        # current_Q = self.critic(state, action, squeeze=False, bs = self.batch_size)
        
        # Compute the target Q value
        target_Q = self.critic_target(
            next_state,
            self.actor_target(state=next_state, bs=self.batch_size, is_linear=True),
             bs=self.batch_size, is_linear=True)
        target_Q = reward + (not_done * self.discount * target_Q).detach()
        # Get current Q estimate
        current_Q = self.critic(state, action,  bs = self.batch_size, is_linear=True)

        # Compute critic loss
        #TODO: Loss is different from jiandanshixian
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        
        # for j in state_loader:
        #     # Compute actor loss
        #     actor_loss = -self.critic(
        #     state, self.actor(j.x,j.edge_index,squeeze=False,bs=self.batch_size), bs=self.batch_size).mean()
        
        # Compute actor loss
        actor_loss = -self.critic(
            state, self.actor(state=state,bs=self.batch_size, is_linear=True), bs=self.batch_size, is_linear=True).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        

        # Update the frozen target models
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss, actor_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        
    def act(self, x, edge_index):
        a0 = self.actor(x, edge_index).squeeze(0).detach().numpy()
        return a0
    
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