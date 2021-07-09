#!/usr/bin/python3
import numpy as np
from util.data_process import *


def show():
    folder = 'saved/'
    episodes = 10000
    actions = 100
    k = 10
    experiment = 'InvManagement-v1'
    v = 3
    id = 0

    # name = 'D:/IC/indProject/coding/wolpertinger/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces-master/results/obj/{}data_{}_Wolp{}_{}{}k{}#{}.json.zip'.format(folder,
    #                                                                  episodes,
    #                                                                  v,
    #                                                                  experiment[:3],
    #                                                                  actions,
    #                                                                  k,
    #                                                                  id
    #                                                                  )
    name="D:\IC\indProject\coding\wolpertinger\Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces-master\results\obj\saved\data_600_Wolp3_Inv1000k100#0.json.zip"
    data_process = Data_handler(name)

    print("Data file is loaded")

    data_process.plot_rewards()
    data_process.plot_average_reward()
    data_process.plot_action_distribution()
    data_process.plot_action_distribution_over_time()
    data_process.plot_action_error()


if __name__ == '__main__':
    show()
