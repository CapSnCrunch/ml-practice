from manim import *
import numpy as np

class GridVisualization(Scene):
    def construct(self):
        # Create sample data (you can replace this with actual Q and C values)
        Q = np.zeros((22, 2, 11, 2))
        C = np.zeros((22, 2, 11, 2))
        
        # Add some sample values for visualization
        Q[11:22, 0, 1:11, 0] = np.random.rand(11, 10)  # Sample values for player sum 11-21, no ace, dealer 1-10, no usable ace
        C[11:22, 0, 1:11, 0] = np.random.rand(11, 10)  # Sample counts
        
        # Create grid for Q-values
        q_grid = self.create_grid(Q, "Usable Ace")
        q_grid.to_edge(LEFT)
        
        # Create grid for C-values
        c_grid = self.create_grid(C, "No Usable Ace")
        c_grid.to_edge(RIGHT)
        
        # Add title
        title = Text("Blackjack State-Action Values", font_size=36)
        title.to_edge(UP, buff=0.5)
        
        # Animate
        self.play(Write(title))
        self.play(Create(q_grid))
        self.play(Create(c_grid))
        self.wait(2)
        
    def create_grid(self, data, title):
        grid = VGroup()
        rows, cols = 11, 10  # For player sum 11-21 and dealer 1-10
        cell_size = 0.5
        
        # Get the min and max values for normalization
        values = data[11:22, 0, 1:11, 0]
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Create grid cells
        for i in range(rows):
            for j in range(cols):
                value = data[i+11, 0, j+1, 0]
                if max_val != min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0
                filled_square = Square(side_length=0.4)
                filled_square.set_fill(BLUE, opacity=normalized_value)
                filled_square.set_stroke(width=0)
                # Centering: y = (rows/2 - 0.5) * cell_size - i * cell_size
                y = (rows/2 - 0.5) * cell_size - i * cell_size
                x = j * cell_size - (cols/2 - 0.5) * cell_size
                filled_square.move_to([x, y, 0])
                grid.add(filled_square)
        
        # Add title
        grid_title = Text(title, font_size=24)
        grid_title.next_to(grid, UP)
        grid.add(grid_title)
        
        # Add player sum labels (11-21)
        player_sums = VGroup()
        for i in range(rows):
            label = Text(str(i+11), font_size=16)
            y = (rows/2 - 0.5) * cell_size - i * cell_size
            label.move_to([-(cols/2) * cell_size - 0.3, y, 0])
            player_sums.add(label)
        grid.add(player_sums)
        
        # Add dealer card labels (1-10)
        dealer_cards = VGroup()
        for j in range(cols):
            label = Text(str(j+1), font_size=16)
            x = j * cell_size - (cols/2 - 0.5) * cell_size
            label.move_to([x, -(rows/2) * cell_size - 0.3, 0])
            dealer_cards.add(label)
        grid.add(dealer_cards)
        
        return grid
