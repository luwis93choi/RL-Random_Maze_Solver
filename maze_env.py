import turtle as t

import random
import numpy as np
import copy

from maze_drawer import Maze_drawer
from maze_generator import Maze_generator

import time
import cv2 as cv
import math

class Maze():

    def __init__(self, height, width, detection_range=2, obstacle_occupancy_prob=0.3):

        # Init Maze with random 0 and 1 distribution
        self.height = height
        self.width = width

        # Init Agent's starting point
        self.curr_agent_pose = [0, 0]
        self.prev_point_val = 0

        # Init Target point
        self.target_pose = [self.height-1, self.width-1]

        # Build random maze based on occupancy probability
        self.obstacle_occupancy_prob = obstacle_occupancy_prob
        self.maze_generator = Maze_generator(height=self.height, width=self.width, agent_pose=self.curr_agent_pose, target_pose=self.target_pose, obstacle_occupancy_prob=self.obstacle_occupancy_prob)      # Generate valid maze using A star

        self.maze, self.curr_agent_pose, self.target_pose, self.optimal_path = self.maze_generator.generate()

        # Draw the maze
        self.maze_drawer = Maze_drawer(self.maze)

        # Init empty local map
        self.local_map = 0.5 * np.ones([self.maze.shape[0], self.maze.shape[1]])
        self.local_map[self.curr_agent_pose[0], self.curr_agent_pose[1]] = 1
        self.local_map[self.target_pose[0], self.target_pose[1]] = 2

        # Build initial local map
        self.detection_range = detection_range
        for i in range(-1 * (self.detection_range), self.detection_range + 1):
            for j in range(-1 * (self.detection_range), self.detection_range + 1):

                if (i == 0) and (j == 0): 
                
                    self.local_map[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 1

                elif self.within_bounds(self.curr_agent_pose[0] + i, 0, self.maze.shape[0]) and self.within_bounds(self.curr_agent_pose[1] + j, 0, self.maze.shape[1]):
                
                    if self.maze[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] == 3:
                
                        self.local_map[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 3

        self.reward = 0.0
        self.done = 0
        self.collision_count = 0

        self.dy = abs(self.target_pose[0] - self.curr_agent_pose[0])
        self.dx = abs(self.target_pose[1] - self.curr_agent_pose[1])

        self.shortest_dy = self.dy
        self.shortest_dx = self.dx

        states = self.local_map.flatten()

        self.maze_drawer.update_maze(self.local_map, curr_pose=self.curr_agent_pose)

    def reset(self):

        # Init Agent's starting point
        self.curr_agent_pose = [0, 0]
        self.prev_point_val = 0

        # Init Target point
        self.target_pose = [self.height-1, self.width-1]

        # Build random maze based on occupancy probability
        self.maze_generator = Maze_generator(height=self.height, width=self.width, agent_pose=self.curr_agent_pose, target_pose=self.target_pose, obstacle_occupancy_prob=self.obstacle_occupancy_prob)      # Generate valid maze using A star

        self.maze, self.curr_agent_pose, self.target_pose, self.optimal_path = self.maze_generator.generate()

        # Init empty local map
        self.local_map = 0.5 * np.ones([self.maze.shape[0], self.maze.shape[1]])
        self.local_map[self.curr_agent_pose[0], self.curr_agent_pose[1]] = 1
        self.local_map[self.target_pose[0], self.target_pose[1]] = 2

        # Build initial local map
        for i in range(-1 * (self.detection_range), self.detection_range + 1):
            for j in range(-1 * (self.detection_range), self.detection_range + 1):

                if (i == 0) and (j == 0): 
                
                    self.local_map[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 1

                elif self.within_bounds(self.curr_agent_pose[0] + i, 0, self.maze.shape[0]) and self.within_bounds(self.curr_agent_pose[1] + j, 0, self.maze.shape[1]):
                
                    if self.maze[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] == 3:
                
                        self.local_map[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 3
                        
        self.reward = 0.0
        self.done = 0
        self.collision_count = 0

        states = self.local_map.flatten()

        self.maze_drawer.update_maze(self.local_map, curr_pose=self.curr_agent_pose)

        return states

    def within_bounds(self, value, low, high):

        return (low <= value) and (value <= high)

    def maze_update(self, dheight=0, dwidth=0):
        
        skip = False

        if self.within_bounds(self.curr_agent_pose[0] + dheight, 0, self.maze.shape[0]) is False:

            # self.reward += math.sqrt((self.height - self.dy)**2 + (self.width - self.dx)**2)
            
            self.reward += -10
            # self.done = 1
            self.collision_count += 1
            skip = True
            self.maze_drawer.update_maze(self.local_map, curr_pose=self.curr_agent_pose)

        elif self.within_bounds(self.curr_agent_pose[1] + dwidth, 0, self.maze.shape[1]) is False:

            # self.reward += math.sqrt((self.height - self.dy)**2 + (self.width - self.dx)**2)

            self.reward += -10
            # self.done = 1
            self.collision_count += 1
            skip = True
            self.maze_drawer.update_maze(self.local_map, curr_pose=self.curr_agent_pose)

        elif self.maze[self.curr_agent_pose[0] + dheight][self.curr_agent_pose[1] + dwidth] >= 3:

            # self.reward += math.sqrt((self.height - self.dy)**2 + (self.width - self.dx)**2)
            # self.local_map[self.curr_agent_pose[0] + dheight][self.curr_agent_pose[1] + dwidth] += 1
            # self.reward += (-10 - 0.001 * self.local_map[self.curr_agent_pose[0] + dheight][self.curr_agent_pose[1] + dwidth])
            self.reward += -10
            # self.done = 1
            self.collision_count += 1
            skip = True
            self.maze_drawer.update_maze(self.local_map, curr_pose=self.curr_agent_pose)

        # if self.collision_count >= 20:
        #     self.done = 1
        #     skip = True

        if skip: return

        prev_agent_pose = copy.deepcopy(self.curr_agent_pose)

        self.curr_agent_pose[0] += dheight
        self.curr_agent_pose[1] += dwidth

        self.maze[prev_agent_pose[0]][prev_agent_pose[1]] = 0
        self.local_map[prev_agent_pose[0]][prev_agent_pose[1]] = 0
        self.local_map[prev_agent_pose[0]][prev_agent_pose[1]] = self.prev_point_val
        self.prev_point_val = self.local_map[self.curr_agent_pose[0]][self.curr_agent_pose[1]] - 1.0
            
        self.dy = abs(self.target_pose[0] - self.curr_agent_pose[0])
        self.dx = abs(self.target_pose[1] - self.curr_agent_pose[1])

        if self.maze[self.curr_agent_pose[0]][self.curr_agent_pose[1]] == 0:

            if self.local_map[self.curr_agent_pose[0]][self.curr_agent_pose[1]] == 0.5:

                self.reward += math.sqrt((self.height - self.dy)**2 + (self.width - self.dx)**2)

            elif self.local_map[self.curr_agent_pose[0]][self.curr_agent_pose[1]] < 0.5:

                self.reward += self.local_map[self.curr_agent_pose[0]][self.curr_agent_pose[1]]

        self.maze[self.curr_agent_pose[0]][self.curr_agent_pose[1]] = 1
        self.local_map[self.curr_agent_pose[0]][self.curr_agent_pose[1]] = 1
        # self.local_map[self.curr_agent_pose[0]][self.curr_agent_pose[1]] = 0

        self.maze_drawer.update_maze(self.local_map, curr_pose=self.curr_agent_pose)

    def step(self, action):

        self.reward = 0.0
        self.done = 0
        success = False

        if action == 0:
            self.maze_update(dheight = 0, dwidth = 1)
            self.reward += -0.01

        elif action == 1:
            self.maze_update(dheight = 0, dwidth = -1)
            self.reward += -0.01

        elif action == 2:
            self.maze_update(dheight = 1, dwidth = 0)
            self.reward += -0.01

        elif action == 3:
            self.maze_update(dheight = 1, dwidth = 1)
            self.reward += -0.01

        elif action == 4:
            self.maze_update(dheight = 1, dwidth = -1)
            self.reward += -0.01

        elif action == 5:
            self.maze_update(dheight = -1, dwidth = 0)
            self.reward += -0.01

        elif action == 6:
            self.maze_update(dheight = -1, dwidth = 1)
            self.reward += -0.01

        elif action == 7:
            self.maze_update(dheight = -1, dwidth = -1)
            self.reward += -0.01

        # Primary Success Reward
        if ((self.target_pose[0] - self.curr_agent_pose[0]) <= 1) and ((self.target_pose[1] - self.curr_agent_pose[1]) <= 1):
            self.done = 1
            self.reward += 1000
            success = True
            self.maze_drawer.update_maze(self.local_map, curr_pose=self.curr_agent_pose)

        for i in range(-1 * (self.detection_range), self.detection_range + 1):
            for j in range(-1 * (self.detection_range), self.detection_range + 1):

                if (i == 0) and (j == 0): 
                
                    self.local_map[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 1

                elif self.within_bounds(self.curr_agent_pose[0] + i, 0, self.maze.shape[0]) and self.within_bounds(self.curr_agent_pose[1] + j, 0, self.maze.shape[1]):
                
                    if self.maze[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] == 3:
                
                        if self.local_map[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] < 3:

                            self.local_map[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 3

        states = self.local_map.flatten()

        # reward_range = np.array(list(range(-30, 10000)))
        # est_reward_mean = reward_range.mean()
        # est_reward_std = reward_range.std()
        # est_reward_max = reward_range.max()
        # est_reward_min =reward_range.min()

        # # self.reward = (self.reward - est_reward_mean) / est_reward_std
        # self.reward = (self.reward - est_reward_min) / (est_reward_max - est_reward_min)
        # # print('Min-Max Reward : {}'.format(self.reward))

        # print(states)
        return states, self.reward, self.done, success
