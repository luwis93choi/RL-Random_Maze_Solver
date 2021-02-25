import cv2 as cv
import numpy as np
import time

class Maze_drawer():

    def __init__(self, maze_grid):

        self.maze_grid = maze_grid

        self.maze_height = maze_grid.shape[0]
        self.maze_width = maze_grid.shape[1]

        self.unit_size = 30

        self.img = np.zeros((self.maze_height * self.unit_size, self.maze_width * self.unit_size, 3), np.uint8)

        self.create_time = time.time()

    def update_maze(self, maze_grid, curr_pose = [0, 0]):

        self.maze_grid = maze_grid

        for i in range(self.maze_height):

            for j in range(self.maze_width):

                if self.maze_grid[i][j] <= 0:

                    self.img[self.unit_size * i : self.unit_size * i + self.unit_size, self.unit_size * j : self.unit_size * j + self.unit_size] = [255, 255, 255]

                elif self.maze_grid[i][j] == 1:
                    
                    self.img[self.unit_size * i : self.unit_size * i + self.unit_size, self.unit_size * j : self.unit_size * j + self.unit_size] = [255, 0, 0]

                elif self.maze_grid[i][j] == 2:
                    
                    self.img[self.unit_size * i : self.unit_size * i + self.unit_size, self.unit_size * j : self.unit_size * j + self.unit_size] = [0, 0, 255]

                elif self.maze_grid[i][j] >= 3:
                   
                    self.img[self.unit_size * i : self.unit_size * i + self.unit_size, self.unit_size * j : self.unit_size * j + self.unit_size] = [0, 0, 0]

                elif self.maze_grid[i][j] == 0.5:
                    
                    self.img[self.unit_size * i : self.unit_size * i + self.unit_size, self.unit_size * j : self.unit_size * j + self.unit_size] = [128, 128, 128]

        cv.imshow('Random Maze [' + str(self.create_time) + ']', self.img)
        cv.waitKey(1)

