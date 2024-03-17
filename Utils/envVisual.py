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

        # Mark wind effect
        if self.environment.wind is not None:
            for col, wind_strength in enumerate(self.environment.wind):
                if wind_strength != 0:
                    arrow_direction = (0, 1)  # Wind direction (up or down)
                    arrow_length = abs(wind_strength) # Wind strength (arrow length)
                    arrow_start = (1, col)  # Arrow starting point (above the grid)
                    arrow_color = (0, 0, 0)  # Black color for arrows
                    plt.arrow(arrow_start[1], arrow_start[0], *arrow_direction, width=0.05, head_width=0.3, head_length=0.3 * arrow_length, color=arrow_color)

                    for row in range(self.environment.rows):
                        grid_color = [0.8, 0.9, 1 - wind_strength / max(self.environment.wind)]
                        grid[row, col] = grid_color




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