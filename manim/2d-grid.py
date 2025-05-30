from manim import *
import numpy as np
import os

min_frame_time = 1 / config["frame_rate"]

class GridVisualization(Scene):
    def construct(self):
        # Load initial policy matrices
        usable_ace_path = os.path.join(os.path.dirname(__file__), '../blackjack/policy_matrix_usable_ace.npy')
        no_ace_path = os.path.join(os.path.dirname(__file__), '../blackjack/policy_matrix_no_ace.npy')
        policy_usable_ace = np.load(usable_ace_path)
        policy_no_ace = np.load(no_ace_path)

        # Load policy update history
        history_path = os.path.join(os.path.dirname(__file__), '../blackjack/policy_update_history.npy')
        history = np.load(history_path, allow_pickle=True)

        # Subsample: take every 300th entry
        history = history[::10000]

        rows, cols = 11, 10  # player sum 11-21, dealer 1-10
        cell_size = 0.5

        # --- Create Usable Ace Grid (left) ---
        grid_squares_usable = [[None for _ in range(cols)] for _ in range(rows)]
        grid_usable = VGroup()
        min_val_usable = np.min(policy_usable_ace)
        max_val_usable = np.max(policy_usable_ace)
        for i in range(rows):
            for j in range(cols):
                value = policy_usable_ace[i, j]
                # opacity = (value - min_val_usable) / (max_val_usable - min_val_usable) if max_val_usable != min_val_usable else 0
                opacity = 0
                square = Square(side_length=0.4)
                square.set_fill(BLUE, opacity=opacity)
                square.set_stroke(width=0)
                y = (rows/2 - 0.5) * cell_size - i * cell_size
                x = j * cell_size - (cols/2 - 0.5) * cell_size
                square.move_to([x, y, 0])
                grid_usable.add(square)
                grid_squares_usable[i][j] = square
        grid_usable.to_edge(LEFT)

        # --- Create No Usable Ace Grid (right) ---
        grid_squares_no = [[None for _ in range(cols)] for _ in range(rows)]
        grid_no = VGroup()
        min_val_no = np.min(policy_no_ace)
        max_val_no = np.max(policy_no_ace)
        for i in range(rows):
            for j in range(cols):
                value = policy_no_ace[i, j]
                # opacity = (value - min_val_no) / (max_val_no - min_val_no) if max_val_no != min_val_no else 0
                opacity = 0
                square = Square(side_length=0.4)
                square.set_fill(BLUE, opacity=opacity)
                square.set_stroke(width=0)
                y = (rows/2 - 0.5) * cell_size - i * cell_size
                x = j * cell_size - (cols/2 - 0.5) * cell_size
                square.move_to([x, y, 0])
                grid_no.add(square)
                grid_squares_no[i][j] = square
        grid_no.to_edge(RIGHT)

        # Add title
        title = Text("Blackjack Policy Visualization", font_size=36)
        title.to_edge(UP, buff=0.5)

        # Animate
        self.play(Write(title))
        self.add(grid_usable, grid_no)

        # --- Add axis labels and numbers for Usable Ace Grid (left) ---
        player_sum_label = Text("Player", font_size=20).rotate(PI/2)
        player_sum_label.next_to(grid_usable, LEFT, buff=0.5)
        self.add(player_sum_label)
        for i in range(rows):
            label = Text(str(i+11), font_size=16)
            y = (rows/2 - 0.5) * cell_size - i * cell_size
            label.next_to(grid_usable, LEFT, buff=0.1)
            label.shift(UP * (y - grid_usable.get_center()[1]))
            self.add(label)
        dealer_label = Text("Dealer", font_size=20)
        dealer_label.next_to(grid_usable, DOWN, buff=0.5)
        self.add(dealer_label)
        for j in range(cols):
            label = Text(str(j+1), font_size=16)
            x = j * cell_size - (cols/2 - 0.5) * cell_size
            label.move_to(grid_usable.get_bottom() + RIGHT * x + DOWN * 0.2)
            self.add(label)

        # --- Add axis labels and numbers for No Usable Ace Grid (right) ---
        player_sum_label_no = Text("Player", font_size=20).rotate(PI/2)
        player_sum_label_no.next_to(grid_no, LEFT, buff=0.5)
        self.add(player_sum_label_no)
        for i in range(rows):
            label = Text(str(i+11), font_size=16)
            y = (rows/2 - 0.5) * cell_size - i * cell_size
            label.next_to(grid_no, LEFT, buff=0.1)
            label.shift(UP * (y - grid_no.get_center()[1]))
            self.add(label)
        dealer_label_no = Text("Dealer", font_size=20)
        dealer_label_no.next_to(grid_no, DOWN, buff=0.5)
        self.add(dealer_label_no)
        for j in range(cols):
            label = Text(str(j+1), font_size=16)
            x = j * cell_size - (cols/2 - 0.5) * cell_size
            label.move_to(grid_no.get_bottom() + RIGHT * x + DOWN * 0.2)
            self.add(label)

        # --- Animate updates from history ---
        for entry in history:
            player_v, usable_ace, dealer_showing, probs = entry
            i = int(player_v) - 11
            j = int(dealer_showing) - 1
            if 0 <= i < rows and 0 <= j < cols:
                opacity = float(probs[1]) if len(probs) > 1 else float(probs[0])
                if usable_ace:
                    self.play(grid_squares_usable[i][j].animate.set_fill(BLUE, opacity=opacity), run_time=min_frame_time)
                else:
                    self.play(grid_squares_no[i][j].animate.set_fill(BLUE, opacity=opacity), run_time=min_frame_time)

        self.wait(2)

    def create_grid(self, data, title):
        grid = VGroup()
        rows, cols = data.shape
        cell_size = 0.5

        # Normalize values for opacity
        min_val = np.min(data)
        max_val = np.max(data)

        for i in range(rows):
            for j in range(cols):
                value = data[i, j]
                if max_val != min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0
                filled_square = Square(side_length=0.4)
                filled_square.set_fill(BLUE, opacity=normalized_value)
                filled_square.set_stroke(width=0)
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
