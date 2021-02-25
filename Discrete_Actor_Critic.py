import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorCriticNetowrk(nn.Module):
    
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(ActorCriticNetowrk, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.actor_policy_net = nn.Linear(self.fc2_dims, n_actions)

        self.critic_value_net = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, observation):

        state = torch.Tensor(observation).to(self.device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        policy = self.actor_policy_net(x)

        value = self.critic_value_net(x)

        return policy, value

class ACAgent():

    def __init__(self, alpha, input_dims, gamma=0.999, 
                layer1_size=600, layer2_size=600, n_actions=8):

        self.gamma = gamma
        self.actor_critic = ActorCriticNetowrk(alpha, input_dims, layer1_size, layer2_size, n_actions, device='cuda:3')

        self.log_probs = None

    def choose_action(self, observation):

        policy, _ = self.actor_critic.forward(observation)
        policy = F.softmax(policy)

        action_probs = torch.distributions.Categorical(policy)
        action = action_probs.sample()

        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, state_, done):

        self.actor_critic.optimizer.zero_grad()

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)
        delta = reward + self.gamma * critic_value_ * (1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()


from maze_env import Maze
import os
import time
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)

if __name__ == '__main__':

    env = Maze(height=21, width=21, detection_range=1, obstacle_occupancy_prob=0.5)

    agent = ACAgent(alpha=0.00001, gamma=0.999, input_dims=[529], n_actions=8, layer1_size=2048, layer2_size=512)

    score_history = []
    avg_score_history = []

    epsiodes = 5000000
    max_steps = 23 * 23

    start_time = str(time.time())

    plt.figure(figsize=(10, 8))

    for e in range(epsiodes):
        
        score = 0
        done = False
        observation = env.reset()

        # while not done:
        for i in range(max_steps):

            action = agent.choose_action(observation)

            observation_, reward, done, success = env.step(action)

            agent.learn(observation, reward, observation_, done)

            observation = observation_

            score += reward
            
            if done:
                break

        score_history.append(score)

        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        print('Epsidoe : ', e, ' | Score : %.2f' % score, ' | Average Score : %.2f' % avg_score)

        if e == 0:
            if os.path.exists('./' + start_time) == False:
                print('Creating save directory')
                os.mkdir('./' + start_time)

        plt.plot([i for i in range(len(score_history))], score_history, 'bo-')
        plt.plot([i for i in range(len(avg_score_history))], avg_score_history, 'r-', linewidth=4)
        plt.title('RL Random Maze Training [Actor-Critic]')
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
