import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from collections import deque

import random
import pickle
import time
import os

from maze_env import Maze

env = Maze(height=21, width=21, detection_range=1)
np.random.seed(42)

class DDQN():

    def __init__(self, action_space, state_space, device):

        self.action_space = action_space
        self.state_space = state_space

        self.NN_hidden = 256

        self.epsilon = 1
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.999

        self.gamma = 0.95
        
        self.batch_size = 128
        self.learning_rate = 0.001

        self.memory = deque(maxlen=100000)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.NN_model_policy_net = nn.Sequential(

            nn.Linear(in_features=self.state_space, out_features=self.NN_hidden),
            nn.BatchNorm1d(num_features=self.NN_hidden),
            nn.ReLU(),

            nn.Linear(in_features=self.NN_hidden, out_features=self.NN_hidden),
            nn.BatchNorm1d(num_features=self.NN_hidden),
            nn.ReLU(),

            nn.Linear(in_features=self.NN_hidden, out_features=self.NN_hidden),
            nn.BatchNorm1d(num_features=self.NN_hidden),
            nn.ReLU(),

            nn.Linear(in_features=self.NN_hidden, out_features=self.action_space),

        ).to(self.device)

        self.NN_model_target_net = nn.Sequential(

            nn.Linear(in_features=self.state_space, out_features=self.NN_hidden),
            nn.BatchNorm1d(num_features=self.NN_hidden),
            nn.ReLU(),

            nn.Linear(in_features=self.NN_hidden, out_features=self.NN_hidden),
            nn.BatchNorm1d(num_features=self.NN_hidden),
            nn.ReLU(),

            nn.Linear(in_features=self.NN_hidden, out_features=self.NN_hidden),
            nn.BatchNorm1d(num_features=self.NN_hidden),
            nn.ReLU(),

            nn.Linear(in_features=self.NN_hidden, out_features=self.action_space),

        ).to(self.device)

        self.loss = nn.MSELoss()

        self.optimizer = optim.Adam(self.NN_model_policy_net.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):

        self.memory.append([state, action, reward, next_state, done])

    def act(self, states):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        self.NN_model_policy_net.eval()
        # with torch.no_grad():
        states = torch.from_numpy(states).float().to(self.device)
        action_Q_values = self.NN_model_policy_net(states)
        
        return np.argmax(action_Q_values.clone().detach().cpu().numpy()[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            print('Accumulating Replay Memory')
            return
        
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        #################################################################################################
        self.NN_model_policy_net.eval()
        future_action_prediction = (self.NN_model_policy_net(next_states)).clone().detach().cpu().numpy()
        max_future_reward_prediction = np.amax(future_action_prediction, axis=1)

        targets = rewards + self.gamma * (max_future_reward_prediction) * (1 - dones)

        targets_full = (self.NN_model_target_net(states)).clone().detach().cpu().numpy()

        idx = np.array([i for i in range(self.batch_size)])
        targets_full[[idx], [actions]] = targets
        #################################################################################################

        #################################################################################################
        self.NN_model_policy_net.train()
        self.optimizer.zero_grad()
        output_Q_values = self.NN_model_policy_net(states)
        loss = self.loss(output_Q_values, torch.from_numpy(targets_full).float().to(self.device))
        loss.backward()
        self.optimizer.step()
        #################################################################################################

        del future_action_prediction
        del max_future_reward_prediction
        del targets_full
        del targets

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

def train_dqn(episode):

    reward_list = []

    avg_reward_list = []
    unit_len = 10

    action_space = 8
    state_space = 10

    max_steps = 21 * 21 * 3

    agent = DDQN(action_space, state_space, device='cuda:3')
    target_update = 10

    success_num = 0
    fail_num = 0
    
    start_time = str(time.time())

    plt.figure(figsize=(10, 8))

    for e in range(episode):

        state = env.reset()
        state = np.reshape(state, (1, state_space))
        score = 0

        # while True:
        for step in range(max_steps):

            action = agent.act(state)

            reward, next_state, done, success = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            agent.replay()

            if done:
                break

        if success: success_num += 1
        else: fail_num += 1

        if success_num == 5000:
            agent.epsilon_min = 0.1

        elif success_num == 10000:
            agent.epsilon_min = 0.01

        print('Episode {}/{} | Score : {:.2f} | Success : {} | Fail : {}'.format(e, episode, score, success_num, fail_num))

        reward_list.append(score)

        if e < unit_len:
            avg_reward_list.append(sum(reward_list[0:0+e+1])/(e+1))
        else:
            avg_reward_list.append(sum(reward_list[e-unit_len+1:e+1])/unit_len)

        # avg_reward_list.append(sum(reward_list)/len(reward_list))

        if e == 0:
            if os.path.exists('./' + start_time) == False:
                print('Creating save directory')
                os.mkdir('./' + start_time)

        plt.plot([i for i in range(len(reward_list))], reward_list, 'bo-')
        plt.plot([i for i in range(len(avg_reward_list))], avg_reward_list, 'r-', linewidth=4)
        plt.title('RL Random Maze Training [DDQN - Hidden ' + str(agent.NN_hidden) +' | 4 FC] \n' + 'Min Epsilon : ' + str(agent.epsilon_min) + '\n' + 'Epsilon Decay : '  + str(agent.epsilon_decay) + '\n' + 'Obstacle Occupancy Probabilty : ' + str(env.maze_generator.obstacle_occupancy_prob))
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.savefig('./' + start_time + '/Training_Result.png')
        plt.cla()

        with open('./' + start_time + '/Reward.txt', 'wb') as reward_list_file:
            pickle.dump(reward_list, reward_list_file)

        if success:
            torch.save({'epoch' : e,
                        'policy_model_state_dict' : agent.NN_model_policy_net.state_dict(),
                        'target_model_state_dict' : agent.NN_model_target_net.state_dict(),
                        'optimizer' : agent.optimizer.state_dict(),
                        'reward' : reward_list}, './' + start_time + '/Random Maze Agent.pth')

        if (e % target_update) == 0:
            agent.NN_model_target_net.load_state_dict(agent.NN_model_policy_net.state_dict())

    return reward_list


if __name__ == '__main__':

    ep = 100000
    loss = train_dqn(ep)