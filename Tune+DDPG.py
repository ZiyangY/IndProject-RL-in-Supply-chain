# -*- coding: utf-8 -*-
import or_gym
from or_gym.utils import create_env
import ray
from ray.rllib import agents
from ray import tune
import time

# env_config = {'dist':6,
#               'dist_param' : {'scale':1,'loc':20}}
# env_config = {'dist':2,
#               'dist_param' : {'n': 20,'p':0.9}}
# env_config = {}

env_config = {'periods': 10,
              'I0':[50],
              'r':[0.75,0.5],
              'k':[0.05, 0.025],
              'h':[0.05],
              'c':[80],
              'L':[10],
              'max_rewards':500}

def register_env(env_name, env_config=env_config):
    env = create_env(env_name)
    tune.register_env(env_name,
        lambda env_name: env(env_name,
            env_config=env_config))

register_env('InvManagement-v1', env_config)
ray.shutdown()
ray.init(ignore_reinit_error=True)

# scale1tune4--> actor_lr = 1e-5, tau=0.02,buffer_size=500
# lstrt2 --> learning_starts 500 (0,300,500,1000)

config = {
    'env': 'InvManagement-v1',
    'num_workers':8,
    "train_batch_size": 64,
    #"tau":1e-2,
    "tau": 0.001,
    #"actor_lr": tune.grid_search([1e-3,1e-4,8e-4,1e-5,2e-5]),
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "gamma":  0.99,
    "buffer_size": 1000000,
    "learning_starts":700,
    #"learning_starts": tune.grid_search([0,300,500,650]),
    
    #Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of iterations.
    "timesteps_per_iteration":10000,
    #"evaluation_interval": 0,
    
    
    "actor_hiddens": [400,300],
    # # Hidden layers activation of the postprocessing stage of the policy
    # # network
    "actor_hidden_activation": "relu",
    # # Postprocess the critic network model output with these hidden layers;
    # # again, if use_state_preprocessor is True, then the state will be
    # # preprocessed by the model specified with the "model" config option first.
    "critic_hiddens":[400,300],
    # # Hidden layers activation of the postprocessing state of the critic.
    "critic_hidden_activation": "relu",
    # # N-step Q learning
    # "n_step": 1,
    
    
    # "exploration_config": {
    #     # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
    #     # actions (after a possible pure random phase of n timesteps).
    #     # "type": "OrnsteinUhlenbeckNoise",
    #     # For how many timesteps should we return completely random actions,
    #     # before we start adding (scaled) noise?
    #     "random_timesteps": 30,
    #     # The OU-base scaling factor to always apply to action-added noise.
    #     # "ou_base_scale": 0.1,
    #     # # The OU theta param.
    #     # "ou_theta": 0.15,
    #     # # The OU sigma param.
    #     # "ou_sigma": 0.2,
    #     # # The initial noise scaling factor.
    #     # "initial_scale": 1.0,
    #     # # The final noise scaling factor.
    #     # "final_scale": 1.0,
    #     # # Timesteps over which to anneal scale (from initial to final values).
    #     # "scale_timesteps": 10000,
    # },
}
stop = {'training_iteration':70}

st=time.time() 
results = tune.run(
    #"DDPG",
    agents.ddpg.DDPGTrainer, # Specify the algorithm to train
    config=config, restore=None, stop=stop,
    checkpoint_freq=1, checkpoint_at_end=True, verbose=1,
    local_dir="./results/default/wolpParm/hidden"
)

print('elapsed time=',time.time()-st)
 
ray.shutdown()
'''
#
# self.periods = 30
#         self.I0 = [100, 100, 200]
#         self.p = 2
#         self.r = [1.5, 1.0, 0.75, 0.5]
#         self.k = [0.10, 0.075, 0.05, 0.025]
#         self.h = [0.15, 0.10, 0.05]
#         self.c = [100, 90, 80]
#         self.L = [3, 5, 10]
#         self.backlog = True
#         self.dist = 1
#         self.dist_param = {'mu': 10}
#         self.alpha = 0.97
#         self.seed_int = 0
#         self.user_D = np.zeros(self.periods)
#         self._max_rewards = 2000
#

# # Environment and RL Configuration Settings
# env_name = 'InvManagement-v1'

# env_config = {'periods': 10,
#               'I0':[50],
#               'r':[0.75,0.5],
#               'k':[0.05, 0.025],
#               'h':[0.05],
#               'c':[80],
#               'L':[10],
#               'max_rewards':500}
# # env_config = {}

# def register_env(env_name, env_config=env_config):
#     env = create_env(env_name)
#     tune.register_env(env_name,
#         lambda env_name: env(env_name,
#             env_config=env_config))


# # Register environment
# register_env(env_name, env_config)


# # Initialize Ray and Build Agent
# ray.init(ignore_reinit_error=True)
# # agent = agents.ppo.PPOTrainer(env=env_name, config=rl_config)
# #agent = agents.ddpg.DDPGTrainer(env=env_name, config=rl_config)

# config = {"env":env_name,
#           "env_config":env_config,
#           "model":dict(
#         vf_share_layers=False,
#         fcnet_activation='elu',
#         fcnet_hiddens=[256, 256]),
#         "lr": tune.grid_search([1e-4, 1e-5, 1e-6]),
#         "train_batch_size": tune.grid_search([2000, 2500, 3000])}
# results = tune.run("PPO",    
#                     stop={'timesteps_total': 2000},
#                     config=config,
#                     local_dir="./results", name="derectPPO")
# ray.shutdown()
'''