import turtle as t

import random
import numpy as np
import copy

from maze_drawer import Maze_drawer
from maze_generator import Maze_generator

import time

import math

class Maze():

    def __init__(self, height, width, detection_range=2):

        # Init Maze with random 0 and 1 distribution
        self.height = height
        self.width = width

        # Init Agent's starting point as the center of maze
        # self.curr_agent_pose = [int(self.height/2), int(self.width/2)]
        # self.prev_agent_pose = [int(self.height/2), int(self.width/2)]
        self.curr_agent_pose = [0, 0]
        self.prev_agent_pose = [0, 0]

        self.obstacle_occupancy_prob = 0.5
        self.maze_generator = Maze_generator(height=self.height, width=self.width, obstacle_occupancy_prob=self.obstacle_occupancy_prob)      # Generate valid maze using A star

        self.maze, self.target_pose, self.optimal_path = self.maze_generator.generate()

        self.maze[self.curr_agent_pose[0]][self.curr_agent_pose[1]] = -1    # Place Agent in the maze

        self.maze[self.target_pose[0]][self.target_pose[1]] = 2         # Place Target location in the maze

        # Draw the maze
        self.maze_drawer = Maze_drawer(self.maze)

        # print(self.maze)

        self.reward = 0
        self.done = 0

        states = []

        self.dy = self.target_pose[0] - self.curr_agent_pose[0]
        self.dx = self.target_pose[1] - self.curr_agent_pose[1]

        self.prev_dy = self.dy
        self.prev_dx = self.dx

        self.shortest_dy = self.dy
        self.shortest_dx = self.dx

        states = [self.dy, self.dx, self.curr_agent_pose[0], self.curr_agent_pose[1]]

        self.visited = np.zeros([self.height, self.width])
        
        self.detection_range = detection_range
        for i in range(-1 * (self.detection_range), self.detection_range + 1):
            for j in range(-1 * (self.detection_range), self.detection_range + 1):
                if (i == 0) and (j == 0): 
                    self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] += -1
                elif (0 <= self.curr_agent_pose[0] + i) and (self.curr_agent_pose[0] + i < self.height) and (0 <= self.curr_agent_pose[1] + j) and (self.curr_agent_pose[1] + j < self.width):
                    # states.append(self.maze[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j])

                    if self.maze[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] == 1:
                        self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 1
                    else:
                        self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] += -1
                # else:
                #     states.append(1)
                    
        states = np.concatenate((states, self.visited.flatten()))
        # print(states)

        self.maze_drawer.update_maze(self.visited, curr_pose=self.curr_agent_pose)

    def reset(self):

        # Init Agent's starting point as the center of maze
        # self.curr_agent_pose = [int(self.height/2), int(self.width/2)]
        # self.prev_agent_pose = [int(self.height/2), int(self.width/2)]
        self.curr_agent_pose = [0, 0]
        self.prev_agent_pose = [0, 0]

        self.maze_generator = Maze_generator(height=self.height, width=self.width, obstacle_occupancy_prob=self.obstacle_occupancy_prob)      # Generate valid maze using A star

        self.maze, self.target_pose, self.optimal_path = self.maze_generator.generate()

        self.maze[self.curr_agent_pose[0]][self.curr_agent_pose[1]] = -1    # Place Agent in the maze

        self.maze[self.target_pose[0]][self.target_pose[1]] = 2         # Place Target location in the maze

        # print(self.maze)

        self.reward = 0
        self.done = 0

        states = []

        self.dy = self.target_pose[0] - self.curr_agent_pose[0]
        self.dx = self.target_pose[1] - self.curr_agent_pose[1]

        self.prev_dy = self.dy
        self.prev_dx = self.dx

        states = [self.dy, self.dx, self.curr_agent_pose[0], self.curr_agent_pose[1]]
        
        self.visited = np.zeros([self.height, self.width])
        
        for i in range(-1 * (self.detection_range), self.detection_range + 1):
            for j in range(-1 * (self.detection_range), self.detection_range + 1):
                if (i == 0) and (j == 0):
                    self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] += -1
                elif (0 <= self.curr_agent_pose[0] + i) and (self.curr_agent_pose[0] + i < self.height) and (0 <= self.curr_agent_pose[1] + j) and (self.curr_agent_pose[1] + j < self.width):
                    if self.maze[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] == 1:
                        self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 1
                    else:
                        self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] += -1
                
        states = np.concatenate((states, self.visited.flatten()))

        self.maze_drawer.update_maze(self.visited, curr_pose=self.curr_agent_pose)
        # print(states)

        return states

    def maze_update(self, dheight=0, dwidth=0):
        
        skip = False

        if ((self.curr_agent_pose[0] + dheight) < 0) or ((self.curr_agent_pose[0] + dheight) >= self.height):

            self.reward += -1000
            # self.done = 1

            skip = True

        elif ((self.curr_agent_pose[1]) + dwidth < 0) or ((self.curr_agent_pose[1] + dwidth) >= self.width):
            
            self.reward += -1000
            # self.done = 1
            
            skip = True

        elif self.maze[self.curr_agent_pose[0] + dheight][self.curr_agent_pose[1] + dwidth] == 1:
            
            self.reward += -1000
            # self.done = 1
            
            skip = True

        if skip: return

        self.prev_dy = copy.deepcopy(self.dy)
        self.prev_dx = copy.deepcopy(self.dx)

        self.prev_agent_pose = copy.deepcopy(self.curr_agent_pose)

        self.curr_agent_pose[0] += dheight
        self.curr_agent_pose[1] += dwidth

        self.dy = self.target_pose[0] - self.curr_agent_pose[0]
        self.dx = self.target_pose[1] - self.curr_agent_pose[1]

        self.maze[self.prev_agent_pose[0]][self.prev_agent_pose[1]] = 0
        
        if self.maze[self.curr_agent_pose[0]][self.curr_agent_pose[1]] == 0:

            if (self.dy < self.shortest_dy) and (self.dx < self.shortest_dx):

                self.reward += (math.sqrt((self.height - self.dy)**2 + (self.width - self.dx)**2) + self.visited[self.curr_agent_pose[0]][self.curr_agent_pose[1]] * math.sqrt(self.dy**2 + self.dx**2) * 0.3)

                self.shortest_dy = copy.deepcopy(self.dy)
                self.shortest_dx = copy.deepcopy(self.dx)

            elif self.visited[self.curr_agent_pose[0]][self.curr_agent_pose[1]] == 0:
                
                self.reward += math.sqrt((self.height - self.dy)**2 + (self.width - self.dx)**2)

            else:

                self.reward += (1 + self.visited[self.curr_agent_pose[0]][self.curr_agent_pose[1]] * 0.3)

        self.maze[self.curr_agent_pose[0]][self.curr_agent_pose[1]] = -1

        # self.visited[self.curr_agent_pose[0]][self.curr_agent_pose[1]] += -1

        self.maze_drawer.update_maze(self.visited, curr_pose=self.curr_agent_pose)

    def step(self, action):

        self.reward = 0
        self.done = 0
        success = False

        if action == 0:
            self.maze_update(dheight = 0, dwidth = 1)
            self.reward += -0.1

        if action == 1:
            self.maze_update(dheight = 0, dwidth = -1)
            self.reward += -0.1

        if action == 2:
            self.maze_update(dheight = 1, dwidth = 0)
            self.reward += -0.1

        if action == 3:
            self.maze_update(dheight = 1, dwidth = 1)
            self.reward += -0.1

        if action == 4:
            self.maze_update(dheight = 1, dwidth = -1)
            self.reward += -0.1

        if action == 5:
            self.maze_update(dheight = -1, dwidth = 0)
            self.reward += -0.1

        if action == 6:
            self.maze_update(dheight = -1, dwidth = 1)
            self.reward += -0.1

        if action == 7:
            self.maze_update(dheight = -1, dwidth = -1)
            self.reward += -0.1

        if self.done != 1:
            # Primary Success Reward
            if (self.dy <= 1) and (self.dx <= 1):
                self.done = 1
                self.reward += 100000
                success = True

            # Secondary Reward
            elif (self.dy <= 2) and (self.dx <= 2):
                # self.done = 1
                self.reward += 1000

        states = [self.dy, self.dx, self.curr_agent_pose[0], self.curr_agent_pose[1]]

        for i in range(-1 * (self.detection_range), self.detection_range + 1):
            for j in range(-1 * (self.detection_range), self.detection_range + 1):
                if (i == 0) and (j == 0):
                    self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] += -1
                elif (0 <= self.curr_agent_pose[0] + i) and (self.curr_agent_pose[0] + i < self.height) and (0 <= self.curr_agent_pose[1] + j) and (self.curr_agent_pose[1] + j < self.width):
                    if self.maze[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] == 1:
                        self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] = 1
                    else:
                        self.visited[self.curr_agent_pose[0] + i][self.curr_agent_pose[1] + j] += -1

        states = np.concatenate((states, self.visited.flatten()))

        return self.reward, states, self.done, success
