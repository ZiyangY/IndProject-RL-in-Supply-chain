#!/usr/bin/python3
import json
import os
from os.path import splitext, basename
import zipfile


def load(file_name):
    data = Data()
    if zipfile.is_zipfile(file_name):
        print('Data: Unziping ', file_name, '...')
        with zipfile.ZipFile(file_name) as myzip:
            string = (myzip.read(myzip.namelist()[0]).decode("utf-8"))
            data.set_data(json.loads(string))
    else:
        print('Data: Loading ', file_name, '...')
        with open(file_name, 'r') as f:
            data.set_data(json.load(f))
    return data


class Data:

    PATH = 'results/obj/'
    AUTOSAVE_BATCH_SIZE = 1e5  # 1 mB

    DATA_TEMPLATE = '''
    {
        "id":0,
        "agent":{
          "name":"default_name",
          "max_actions":0,
          "k":0,
          "version":0
        },
        "experiment":{
          "name":"no_exp",
          "actions_low":null,
          "actions_high":null,
          "number_of_episodes":0
        },
        "simulation":{
          "episodes":[]
        }

    }
    '''

    EPISODE_TEMPLATE = '''
    {
        "id":0,
        "states":[],
        "actions":[],
        "actors_actions":[],
        "ndn_actions":[],
        "rewards":[]
    }
    '''

    def __init__(self):
        self.data = json.loads(self.DATA_TEMPLATE)
        self.episode = json.loads(self.EPISODE_TEMPLATE)
        self.episode_id = 0
        self.temp_saves = 0
        self.data_added = 0

    def __increase_data_counter(self, n=1):
        self.data_added += n

    def set_id(self, n):
        self.data['id'] = n

    def set_agent(self, name, max_actions, k, version):
        self.data['agent']['name'] = name
        self.data['agent']['max_actions'] = max_actions
        self.data['agent']['k'] = k
        self.data['agent']['version'] = version

    def set_experiment(self, name, low, high, eps):
        self.data['experiment']['name'] = name
        self.data['experiment']['actions_low'] = low
        self.data['experiment']['actions_high'] = high
        self.data['experiment']['number_of_episodes'] = eps

    def set_state(self, state):
        self.episode['states'].append(state)
        self.__increase_data_counter(len(state))

    def set_action(self, action):
        self.episode['actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_actors_action(self, action):
        self.episode['actors_actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_ndn_action(self, action):
        self.episode['ndn_actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_reward(self, reward):
        self.episode['rewards'].append(reward)
        self.__increase_data_counter()

    def end_of_episode(self):
        self.data['simulation']['episodes'].append(self.episode)
        self.episode = json.loads(self.EPISODE_TEMPLATE)
        self.episode_id += 1
        self.episode['id'] = self.episode_id

    def finish_and_store_episode(self):
        self.end_of_episode()
        # print(self.data_added / self.AUTOSAVE_BATCH_SIZE)
        if self.data_added > self.AUTOSAVE_BATCH_SIZE:
            self.temp_save()

    def get_file_name(self):
        return 'data_{}_{}_{}{}k{}#{}'.format(self.get_episodes(),
                                              self.get_agent_name(),
                                              self.get_experiment()[:3],
                                              self.data['agent']['max_actions'],
                                              self.data['agent']['k'],
                                              self.get_id())

    def get_episodes(self):
        return self.data['experiment']['number_of_episodes']

    def get_agent_name(self):
        return '{}{}'.format(self.data['agent']['name'][:4],
                             self.data['agent']['version'])

    def get_id(self):
        return self.data['id']

    def get_experiment(self):
        return self.data['experiment']['name']

    def print_data(self):
        print(json.dumps(self.data, indent=2, sort_keys=True))

    def print_stats(self):
        for key in self.data.keys():
            d = self.data[key]
            if key == 'simulation':
                print('episodes:', len(d['episodes']))
            else:
                print(json.dumps(d, indent=2, sort_keys=True))

    def merge(self, data_in):
        if type(data_in) is Data:
            data = data_in.data
        else:
            data = data_in

        for ep in data['simulation']['episodes']:
            self.episode = ep
            self.end_of_episode()

    def set_data(self, data):
        self.data = data

    def save(self, path='', final_save=True):
        if final_save and self.temp_saves > 0:
            if self.data_added > 0:
                self.end_of_episode()
                self.temp_save()
            print('Data: Merging all temporary files')
            for i in range(self.temp_saves):
                file_name = 'D:/IC/indProject/coding/wolpertinger/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces-master/results/obj/temp/{}{}.json'.format(
                                                      i,
                                                      self.get_file_name())
                # print(file_name)
                temp_data = load(file_name)
                # temp_data.print_data()
                # print('^^^^^^^^^^^^')
                self.merge(temp_data)
                os.remove(file_name)

        final_file_name = "D:/IC/indProject/coding/wolpertinger/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces-master/results/obj/" + path + self.get_file_name() + '.json'
        if final_save:
            print('Data: Ziping', final_file_name)
            with zipfile.ZipFile(final_file_name + '.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:
                myzip.writestr(basename(final_file_name), json.dumps(
                    self.data, indent=2, sort_keys=True))
        else:
            with open(final_file_name, mode='w') as f:
                print('Data: Saving', final_file_name)
                json.dump(self.data, f)

    def temp_save(self):
        if self.data_added == 0:
            return
        self.save(path='temp/' + str(self.temp_saves), final_save=False)
        self.temp_saves += 1
        self.data['simulation']['episodes'] = []  # reset
        self.data_added = 0


# if __name__ == '__main__':
#
#     import numpy as np
#     import random
#
#     # d = load('results/obj/saved/data_10001_Wolp3_InvertedPendulum-v1#0.json.zip')
#     # # d = load('results/obj/saved/data_10000_agent_name4_exp_name#0.json.zip')
#     # print(d.get_file_name())
#     # d = load('results/obj/data_10000_agent_name4_exp_name#0.json.zip')
#     d = Data()
#     d.set_agent('agent_name', 1000, 10, 4)
#     d.set_experiment('exp_name', [-2, -3], [3, 2], 10000)
#
#     # d.print_data()
#     #
#     for i in range(10):
#         d.set_state([i, i, i, i])
#         d.set_action([i, i])
#         d.set_actors_action([i, i])
#         d.set_ndn_action([i, i])
#         d.set_reward(i)
#         if i % 3 == 0:
#             d.finish_and_store_episode()
#             d.temp_save()
#             # exit()
#
#     # for i in range(30, 400):
#     #     d.set_state([i, i, i, i])
#     #     d.set_action([i, i])
#     #     d.set_actors_action([i, i])
#     #     d.set_ndn_action([i, i])
#     #     d.set_reward(random.randint(0, 10))
#     #     if i % 2 == 0:
#     #         d.finish_and_store_episode()
#     #         d.temp_save()
#     # #
#     d.temp_save()
#     d.temp_save()
#     d.save()
#     d.print_data()
