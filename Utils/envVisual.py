import numpy as np
import matplotlib.pyplot as plt

class EnvironmentVisualizer:
    def __init__(self, environment):
        self.environment = environment

    def visualize(self):
        # Mark start and goal positions
        grid = np.zeros((self.environment.rows, self.environment.cols, 3))  # RGB grid

        # Mark start and goal positions
        start_row, start_col = self.environment.start
        goal_row, goal_col = self.environment.goal
        grid[start_row, start_col] = [0, 1, 0]  # Green for start position
        grid[goal_row, goal_col] = [1, 0, 0]    # Red for goal position

        # Mark current position if not None
        if self.environment.current_state is not None:
            current_row, current_col = self.environment.current_state
            grid[current_row, current_col] = [0, 0, 1]  # Blue for current position

        # Create the plot
        plt.imshow(grid, cmap='gray', origin='upper', interpolation='none')

        # Add grid lines
        plt.grid(which='both', color='gray', linestyle='-', linewidth=1)

        plt.xticks(np.arange(-0.5, self.environment.cols, 1))
        plt.yticks(np.arange(-0.5, self.environment.rows, 1))

        plt.title('Grid Environment')
        plt.xlabel('Column')
        plt.ylabel('Row')

        plt.colorbar(label='State Value')

        plt.show()