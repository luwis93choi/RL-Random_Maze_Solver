from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import numpy as np

import copy

import random

random.seed(42)

class Maze_generator():

    def __init__(self, height, width, agent_pose, target_pose, obstacle_occupancy_prob=0.2):

        # Init Maze with random 0 and 1 distribution
        self.height = height
        self.width = width

        # Init Agent's starting point as the center of maze
        self.agent_pose = agent_pose
        self.target_pose = target_pose

        self.obstacle_occupancy_prob = obstacle_occupancy_prob

    def generate(self):

        while True:

            self.maze = np.zeros([self.height, self.width])

            for i in range(self.height):
                for j in range(self.width):
                    if np.random.rand() <= self.obstacle_occupancy_prob:
                        self.maze[i][j] = 1

            temp_maze = 1-copy.deepcopy(self.maze)
           
            grid = Grid(matrix=temp_maze)

            start = grid.node(self.agent_pose[0], self.agent_pose[1])
            end = grid.node(self.target_pose[0], self.target_pose[1])

            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path, runs = finder.find_path(start, end, grid)

            if len(path) > 0: break

        self.maze = 3 * self.maze
        self.maze[self.agent_pose[0]][self.agent_pose[1]] = 1
        self.maze[self.target_pose[0]][self.target_pose[1]] = 2

        self.top_bottom_wall = 3 * np.ones([1, self.maze.shape[1]])

        self.maze = np.vstack((self.top_bottom_wall, self.maze, self.top_bottom_wall))

        self.left_right_wall = 3 * np.ones([self.maze.shape[0], 1])

        self.maze = np.hstack((self.left_right_wall, self.maze, self.left_right_wall))

        self.target_pose = [self.target_pose[0] + 1, self.target_pose[1] + 1]
        self.agent_pose = [self.agent_pose[0] + 1, self.agent_pose[1] + 1]

        return self.maze, self.agent_pose, self.target_pose, path