from . import environment
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

class Environment(environment.BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    # def __init__(self):


    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.rows = 10
        self.cols = 10
        self.start = [9,0]
        self.goal = [0,9]
        self.current_state = None

        self.wind = [0, 0, 0, 2, 1, 1, 2, 0, 0, 0]
        self.obstacles = [[1,9], [1,8], [2,8], [3,8], [1,1], [2,2], [3,3]]
        self.mines = [[8,2], [8,3], [8,4], [8,5], [8,6], [8,7], [6,2], [6,3], [6,4], [6,5], [6,6], [6,7]]

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start  # An empty NumPy array

        self.reward_obs_term = (0.0, self.observation(self.current_state), False)

        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        new_state = deepcopy(self.current_state)
        
        if action == 0: #right
            new_state[1] = min(new_state[1]+1, self.cols-1)
        elif action == 1: #down
            new_state[0] = max(new_state[0]-1, 0)
        elif action == 2: #left
            new_state[1] = max(new_state[1]-1, 0)
        elif action == 3: #up
            new_state[0] = min(new_state[0]+1, self.rows-1)
        else:
            raise Exception("Invalid action.")
        reward = -1.0
        is_terminal = False

        # if next position is blocked, stay in the same position
        if new_state in self.obstacles:
            new_state = self.current_state

        # Apply wind effect
        wind_effect = self.wind[self.current_state[1]]
        if wind_effect != 0:
            # check if there's any obstacle
            is_obstacle = False
            for pos_x in range(new_state[0], new_state[0] + wind_effect + 1):
                if (pos_x, new_state[1]) in self.obstacles:
                    new_state = (pos_x-1, new_state[1])
                    is_obstacle = True
                    break
            # if no obstacle, check if cross the boarder
            if is_obstacle == False:
                new_state[0] = min(new_state[0] + wind_effect, 9)
        else:
            # if new position is mine, penalize with a large negative reward,
            # reset the agent back to the start position
            if new_state in self.mines:
                reward = -100.0
                new_state = deepcopy(self.start)

        if new_state == self.goal:
            self.current_state = new_state
            is_terminal = True

        self.current_state = new_state

        self.reward_obs_term = (reward, self.observation(self.current_state), is_terminal)

        return self.reward_obs_term

    def observation(self, state):
        return state[0] * self.cols + state[1] 

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"
    
    def render(self):
        """Render the current state of the environment using Matplotlib"""
        grid = np.zeros((self.rows, self.cols))

        # Mark start, goal, and current positions
        start_row, start_col = self.start
        goal_row, goal_col = self.goal
        current_row, current_col = self.current_state
        grid[start_row, start_col] = 0.5  # Start position
        grid[goal_row, goal_col] = 1.0    # Goal position
        grid[current_row, current_col] = 0.8  # Current position

        # Create the plot
        plt.imshow(grid, cmap='gray', origin='upper', interpolation='none')

        # Add grid lines
        plt.grid(which='both', color='gray', linestyle='-', linewidth=1)

        plt.xticks(np.arange(-0.5, self.cols, 1))
        plt.yticks(np.arange(-0.5, self.rows, 1))

        plt.title('Grid Environment')
        plt.xlabel('Column')
        plt.ylabel('Row')

        plt.colorbar(label='State Value')

        plt.show()