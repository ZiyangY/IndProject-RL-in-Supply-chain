#!/usr/bin/python3
import gym
import numpy as np
from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data
from util.timer import Timer
import or_gym
import matplotlib.pyplot as plt
from matplotlib import gridspec

#max_step will be 30 because of the default environment setting.
def run(episodes=6000,
        render=False,
        experiment='InvManagement-v1',
        max_actions=100,
        knn=0.1):

    env_config = {'dist':6,
              'dist_param' : {'scale':2,'loc':20}}
    '''
    env_config = {'periods': 10,
                  'I0':[100,200],
                  'r':[1.00,0.75,0.50],
                  'k':[0.075,0.05, 0.025],
                  'h':[0.10,0.05],
                  'c':[90,80],
                  'L':[5,10],
                  'max_rewards':1000}
    '''
    env = or_gym.make(experiment,env_config=env_config)

    print(env.observation_space)
    print(env.action_space)

    steps = 10000

    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn)

    timer = Timer()
    
    data = util.data.Data()
    data.set_agent(agent.get_name(), int(agent.action_space.get_number_of_actions()),
                   agent.k_nearest_neighbors, 3)
    data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)
    
    agent.add_data_fetch(data)
    print(data.get_file_name())
    
    full_epoch_timer = Timer()
    reward_sum = 0
    
    rewards = []
    reward_lst = []
    
    # var = 5 # control exploration
    for ep in range(episodes):
        '''
        if ep > DDPGAgent.REPLAY_MEMORY_SIZE:
            var *= .9998    # decay the action randomness
        #print(var)
        '''
        
        timer.reset()
        observation = env.reset()

        total_reward = 0

        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
        for t in range(steps):

            action = agent.act(observation)
            
            # action = np.clip(np.random.normal(action, var),0,max(action)+var)  # add randomness to action selection for exploration
            
            # data.set_action(action.tolist())

            # data.set_state(observation.tolist())

            prev_observation = observation
            observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)

            # data.set_reward(reward)
            

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            agent.observe(episode)

            total_reward += reward
            if done or (t == steps - 1):
                t += 1
                reward_sum += total_reward
                time_passed = timer.get_time()
                rewards.append(reward_sum / (ep + 1))
                reward_lst.append(total_reward)
                print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
                                                                            time_passed, round(
                                                                                time_passed / t),
                                                                            round(reward_sum / (ep + 1))))

                #data.finish_and_store_episode()

                break
    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    # data.save()
    p = 50
    mean_rewards = np.array([np.mean(reward_lst[i-p:i+1]) 
                    if i >= p else np.mean(reward_lst[:i+1]) 
                    for i, _ in enumerate(reward_lst)])
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
    ax0.set_ylabel('Rewards')
    ax0.set_xlabel('Episode')
    ax0.set_title('Training Rewards (ddpg wolpertinger) k-ratio = ' + str(knn)+' norm scale=2 bs=1024')
    ax0.legend()
    plt.ylim([-300, 500])
    plt.grid()

    
    
if __name__ == '__main__':
    knn = [7]
    for i in knn:
        run(knn=0.1*i)
    
    #run(knn=0.6)
    # x = [1,2,3]
    # y = [[1,2,3],[4,5,6],[7,8,9]]
    # plt.xlabel("Episode")
    # plt.ylabel("Rewards")
    # plt.title("A test graph")
    # for i in range(len(rewards[0])):
    #     plt.plot(x,[pt[i] for pt in rewards],label = 'id %s'%i)
    # plt.legend()
    # plt.show()
