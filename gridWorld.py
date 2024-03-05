"""
GridWorld Environment

Reference: https://github.com/qqiang00/reinforce/blob/master/reinforce/gridworld.py
"""

import math
import pygame
import gymnasium as gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Grid(object):
    def __init__(self, x=None, y=None, type=0, reward=0, value=0):
        self.x = x # Coordinate X
        self.y = y # Coordinate Y
        self.type = type # Type (0:empty; 1:obstacle or boundary)
        self.reward = reward # instant reward for an agent entering this grid cell
        self.value = value # the value of this grid cell, for future usage
        self.name = None # name of this grid
        self._update_name()
    
    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)
    
    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(self.x,
                                                                   self.y,
                                                                   self.type,
                                                                   self.reward,
                                                                   self.value,
                                                                   self.name
                                                                    )

class GridMatrix(object):
    def __init__(self, n_width, n_height, default_type=0, default_reward=0, default_value=0):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()
    
    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x, 
                                        y, 
                                        self.default_type, 
                                        self.default_reward, 
                                        self.default_value))
                
    def get_grid(self, x, y):
        '''
        get the information of  a target grid
        '''
        index = y * self.n_width + x
        return self.grids[index]
    
    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise("grid doesn't exist!")
    
    def set_value(self, x, y, value):
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.value = value
        else:
            raise("grid doesn't exist!")

    def set_type(self, x, y, type):
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.type = type
        else:
            raise("grid doesn't exist!")
    
    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type

class GridWorldEnv(gym.Env):
    """
    The grid world environment, inherited from gym.Env
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, n_width=10, n_height=10, u_size=40, default_reward=0, default_type=0, windy=False):
        self.u_size = u_size # size for each cell (pixels)
        self.n_width = n_width
        self.n_height = n_height
        self.width = u_size * n_width # scenario width (pixels)
        self.height = u_size * n_height
        self.default_reward = default_reward
        self.default_type = default_type
        
        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0)
        
        self.reward = 0 # for rendering
        self.action = None # for rendering
        self.windy = windy # whether this is a windy environment

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4) # 0,1,2,3,4 represent left, right, up, down, -, five moves
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)
        self.ends =[(9, 9)] # ending grids, multiple grids allowed
        self.start = (0, 0) # starting girds, only one
        self.types = [] # spcial type of cells, (x,y,z) represents in position (x,y), the cell type is z
        self.rewards = [] # special reward for a cell
        self.refresh_setting()
        self.viewer = None
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.action = action
        old_x, old_y = self._state_to_xy(self.state)
        nex_x, new_y = old_x, old_y

        if action == 0: new_x -= 1   # left
        elif action == 1: new_x += 1   # right
        elif action == 2: new_y += 1   # up
        elif action == 3: new_y -= 1   # down

        elif action == 4: new_x,new_y = new_x-1,new_y-1
        elif action == 5: new_x,new_y = new_x+1,new_y-1
        elif action == 6: new_x,new_y = new_x+1,new_y-1
        elif action == 7: new_x,new_y = new_x+1,new_y+1
        # boundary effect
        if new_x < 0: new_x = 0
        if new_x >= self.n_width: new_x = self.n_width-1
        if new_y < 0: new_y = 0
        if new_y >= self.n_height: new_y = self.n_height-1

        # wall effect, obstacles or boundary.
        # grids with type = 1 are obstacles, cannot be entered
        if self.grids.get_type(new_x,new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)

        done = self._is_end_state(new_x, new_y) 
        self.state = self._xy_to_state(new_x, new_y)
        # provide information with info
        info = {"x":new_x,"y":new_y, "grids":self.grids}
        return self.state, self.reward, done, info
    
    # set status into an one-axis coordinate value
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y
    
    def _xy_to_state(self, x, y):
        return x + self.n_width * y
    
    def refresh_setting(self):
        for x,y,r in self.rewards:
            self.grids.set_reward(x,y,r)
        for x,y,t in self.types:
            self.grids.set_type(x,y,t)
    
    def _reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state
    
    def _is_end_state(self, x, y):
        for end in self.ends:
            if x == end[0] and y == end[1]:
                return True
        return False
    
    # Rendering the grid world with graphic UI
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )