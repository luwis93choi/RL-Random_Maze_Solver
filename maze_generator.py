from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import numpy as np

import copy

import random

random.seed(42)

class Maze_generator():

    def __init__(self, height, width, obstacle_occupancy_prob=0.2):

        # Init Maze with random 0 and 1 distribution
        self.height = height
        self.width = width

        # Init Agent's starting point as the center of maze
        # self.curr_agent_pose = [int(self.height/2), int(self.width/2)]
        # self.prev_agent_pose = [int(self.height/2), int(self.width/2)]
        self.curr_agent_pose = [0, 0]
        self.prev_agent_pose = [0, 0]

        self.obstacle_occupancy_prob = obstacle_occupancy_prob

    def generate(self):

        self.target_pose = [self.height-1, self.width-1]

        while True:

<<<<<<< HEAD
            self.maze = np.zeros([self.height, self.width])
=======
            # self.maze = np.random.randint(low=0, high=2, size=(self.height, self.width))
           
            self.maze = []
            for i in range(self.height):
                self.maze.append([0] * self.width)
            self.maze = np.array(self.maze)
>>>>>>> 87cd37923276b09b4c0273075d02ccb9fca791d0

            for i in range(self.height):
                for j in range(self.width):
                    if np.random.rand() <= self.obstacle_occupancy_prob:
                        self.maze[i][j] = 1

            self.temp_maze = 1-copy.deepcopy(self.maze)
           
            grid = Grid(matrix=self.temp_maze)

            start = grid.node(self.curr_agent_pose[0], self.curr_agent_pose[1])
            end = grid.node(self.target_pose[0], self.target_pose[1])

            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path, runs = finder.find_path(start, end, grid)

            # print(path)
            # print('operations:', runs, 'path length:', len(path))
            # print(grid.grid_str(path=path, start=start, end=end))

            if len(path) > 0: break

        return self.maze, self.target_pose, path