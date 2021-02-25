import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from statistics import mean, stdev

class ActorCriticNetowrk(nn.Module):
    
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(ActorCriticNetowrk, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.FC = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),
        )
        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.actor_policy_net = nn.Sequential(
            nn.Linear(fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, n_actions),
        )

        self.critic_value_net = nn.Sequential(
            nn.Linear(fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.LeakyReLU(),

            nn.Linear(self.fc1_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):

        state = torch.Tensor(observation).to(self.device)

        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))

        x = self.FC(state)

        policy = self.actor_policy_net(x)

        value = self.critic_value_net(x)

        return policy, value

class Intrinsic_Curiosity_Net(nn.Module):
    def __init__(self, lr, input_dims, feature_extractor_dim, foward_model_dim, inverse_model_dim, action_space, device):
        super(Intrinsic_Curiosity_Net, self).__init__()

        self.input_dims = input_dims

        self.feature_extractor = nn.Sequential(
            nn.Linear(*self.input_dims, feature_extractor_dim),
            nn.LayerNorm(feature_extractor_dim),
            nn.LeakyReLU(),

            nn.Linear(feature_extractor_dim, feature_extractor_dim),
            nn.LayerNorm(feature_extractor_dim),
            nn.LeakyReLU(),

            nn.Linear(feature_extractor_dim, feature_extractor_dim),
            nn.LayerNorm(feature_extractor_dim),
            nn.LeakyReLU(),

            nn.Linear(feature_extractor_dim, feature_extractor_dim),
        )

        self.foward_model = nn.Sequential(
            nn.Linear(feature_extractor_dim + action_space, foward_model_dim),
            nn.LayerNorm(foward_model_dim),
            nn.LeakyReLU(),

            nn.Linear(foward_model_dim, foward_model_dim),
            nn.LayerNorm(foward_model_dim),
            nn.LeakyReLU(),

            nn.Linear(foward_model_dim, foward_model_dim),
            nn.LayerNorm(foward_model_dim),
            nn.LeakyReLU(),

            nn.Linear(foward_model_dim, foward_model_dim),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_extractor_dim * 2, inverse_model_dim),
            nn.LayerNorm(inverse_model_dim),
            nn.LeakyReLU(),

            nn.Linear(inverse_model_dim, inverse_model_dim),
            nn.LayerNorm(inverse_model_dim),
            nn.LeakyReLU(),

            nn.Linear(inverse_model_dim, inverse_model_dim),
            nn.LayerNorm(inverse_model_dim),
            nn.LeakyReLU(),

            nn.Linear(inverse_model_dim, inverse_model_dim),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, observation_, policy):

        state = torch.Tensor(observation).to(self.device)
        state_ = torch.Tensor(observation_).to(self.device)

        feature_state = F.tanh(self.feature_extractor(state))
        feature_state_ = F.tanh(self.feature_extractor(state_))
        
        next_feature_state_pred = self.foward_model(torch.cat([feature_state, policy]))
        action_pred = self.inverse_model(torch.cat([feature_state, feature_state_]))

        return next_feature_state_pred, feature_state, feature_state_, action_pred

class AC_Curiosity_Agent():

    def __init__(self, alpha, input_dims, gamma=0.999, 
                layer1_size=600, layer2_size=600, n_actions=8,
                curiosity_feautre_size=600, curiosity_foward_size=600, curiosity_inverse_size=8, 
                forward_ratio=0.2, intrinsic_weight=100, AC_loss_ratio=0.1):

        self.gamma = gamma
        self.actor_critic = ActorCriticNetowrk(alpha, input_dims, layer1_size, layer2_size, n_actions, device='cuda:3')
        self.curiosity_net = Intrinsic_Curiosity_Net(alpha, input_dims, curiosity_feautre_size, curiosity_foward_size, curiosity_inverse_size, n_actions, device='cuda:3')

        self.beta = forward_ratio
        self.intrinsic_weight = intrinsic_weight
        self.AC_loss_ratio = AC_loss_ratio

        self.log_probs = None

        self.delta_stack = deque(maxlen=1024)

        self.foward_mse_loss = nn.MSELoss()
        # self.inverse_mse_loss = nn.MSELoss()
        self.inverse_mse_loss = nn.CrossEntropyLoss()

    def choose_action(self, observation):

        policy, _ = self.actor_critic.forward(observation)
        policy = F.softmax(policy, dim=0)

        action_probs = torch.distributions.Categorical(policy)
        action = action_probs.sample()

        self.log_probs = action_probs.log_prob(action)
        
        return action.item(), policy

    def learn(self, state, reward, state_, policy, action, done):

        self.actor_critic.optimizer.zero_grad()
        self.curiosity_net.optimizer.zero_grad()

        ### Curioisity - Intrinsic Reward Learning ###
        next_feature_state_pred, feature_state, feature_state_, action_pred = self.curiosity_net.forward(state, state_, policy)
        foward_loss = self.foward_mse_loss(next_feature_state_pred, feature_state_)
        
        action_tensor = torch.tensor(action).reshape(1).to(self.actor_critic.device)
        action_pred_reshape = action_pred.reshape(1, 8)

        inverse_loss = self.inverse_mse_loss(action_pred_reshape, action_tensor)
        icm_loss = self.beta * foward_loss + (1-self.beta) * inverse_loss

        intrinsic_reward = self.intrinsic_weight * foward_loss.detach()

        ### AC Learning ###
        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        extrinsic_reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)

        total_reward = extrinsic_reward + intrinsic_reward

        delta = total_reward + self.gamma * critic_value_ * (1-int(done)) - critic_value

        # print('Extrinsic Reward : {:.2f} | Intrinsic Reward : {:.4f} | Delta : {:.2f}'.format(extrinsic_reward, intrinsic_reward, delta.item()))
        # self.delta_stack.append(delta.item())
        # if len(self.delta_stack) > 1:
        #     delta_mean = mean(self.delta_stack)
        #     delta_std = stdev(self.delta_stack)
        #     normalized_delta = (delta - delta_mean) / delta_std

        #     actor_loss = -self.log_probs * normalized_delta
        #     critic_loss = normalized_delta**2
            
        # else:

        #     actor_loss = -self.log_probs * delta
        #     critic_loss = delta**2
        
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (self.AC_loss_ratio * (actor_loss + critic_loss) + icm_loss).backward()
        self.curiosity_net.optimizer.step()
        self.actor_critic.optimizer.step()

        return total_reward

from maze_env import Maze
import os
import time
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)

if __name__ == '__main__':

    env = Maze(height=21, width=21, detection_range=1, obstacle_occupancy_prob=0.4)

    agent = AC_Curiosity_Agent(alpha=0.00001, gamma=0.9, input_dims=[529], n_actions=8, 
                                layer1_size=2048, layer2_size=1024,
                                curiosity_feautre_size=256, curiosity_foward_size=256, curiosity_inverse_size=8, 
                                forward_ratio=0.2, intrinsic_weight=100, AC_loss_ratio=0.1)

    score_history = []
    avg_score_history = []

    epsiodes = 10000000
    max_steps = 23 * 23
    # max_steps = 23 * 23 * 2

    success_num = 0

    start_time = str(time.time())

    plt.figure(figsize=(10, 8))

    for e in range(epsiodes):
        
        score = 0
        done = False
        observation = env.reset()

        # while not done:
        for i in range(max_steps):

            action, policy = agent.choose_action(observation)

            observation_, reward, done, success = env.step(action)

            agent.learn(observation, reward, observation_, policy, action, done)

            observation = observation_

            score += reward
            
            if success:
                success_num += 1

            if done:
                break

        score_history.append(score)

        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        print('Epsidoe : ', e, ' | Score : %.2f' % score, ' | Average Score : %.2f' % avg_score, ' | Success : ', success_num, ' | Fail : ', (e-success_num+1))

        if e == 0:
            if os.path.exists('./' + start_time) == False:
                print('Creating save directory')
                os.mkdir('./' + start_time)

        plt.plot([i for i in range(len(score_history))], score_history, 'bo-')
        plt.plot([i for i in range(len(avg_score_history))], avg_score_history, 'r-', linewidth=4)
        plt.title('RL Random Maze Training [Actor-Critic + Curiosity]')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.savefig('./' + start_time + '/Training_Result.png')
        plt.cla()

        with open('./' + start_time + '/score_history.txt', 'wb') as reward_list_file:
            pickle.dump(score_history, reward_list_file)

        if success:
            torch.save({'epoch' : e,
                        'ActorCritic_model_state_dict' : agent.actor_critic.state_dict(),
                        'optimizer' : agent.actor_critic.optimizer.state_dict(),
                        'reward' : score_history}, './' + start_time + '/Random Maze Agent.pth')