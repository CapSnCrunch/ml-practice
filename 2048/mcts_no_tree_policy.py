import pygame
import numpy as np
from engine import Board, GameWindow

class Node():
    def __init__(self, value = 0):
        self.parent = None
        self.children = {}
        self.value = value

    def create_child(self, action, value):
        child = Node(value)
        child.parent = self
        self.children[action] = child
        return child

    def depth(self):
        if not self.children:
            return 0
        return max([child.depth() for child in self.children.values()]) + 1

    def __str__(self):
        return f'\nValue: {self.value}\n  Parent: {self.parent}\n  Children: {self.children}'

def epsilon_greedy_policy(node, valid_actions, epsilon = 0.01):
    best_value = None
    if node and node.children:
        best_action = max(node.children, key=lambda k: node.children[k].value)
        best_value = node.children[best_action].value

    if best_value is None or np.random.uniform(0, 1) < epsilon:
        # Choose a random node 
        return np.random.choice(valid_actions)
    else:
        # Choose a random node with value = best_value
        best_actions = []
        for (key, child) in node.children.items():
            if child.value == best_value:
                best_actions.append(key)
        return np.random.choice(best_actions)

width, height = 600, 700
thickness = 50

win = pygame.display.set_mode((width, height))
pygame.display.set_caption('2048')

game = GameWindow(4, width, height, thickness)

def redraw_window(win):
    game.draw(win)
    pygame.display.update()

if __name__ == '__main__':
    
    b = Board(4)

    # Random Policy
    random_scores = []
    for i in range(100):
        state, valid_actions = b.restart()
        while True:
            action = epsilon_greedy_policy(None, valid_actions, 1)
            _, _, done, valid_tree_actions = b.step(action)
            if done:
                print(f'Random Policy Score, {b.score}')
                random_scores.append(b.score)
                break
    
    clock = pygame.time.Clock()
 
    # Main loop
    while True:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Monte Carlo Tree Search
        mcts_scores = []
        for i in range(100):
            
            root = Node()
            
            state, valid_actions = game.board.restart()
            
            while True:

                current = root

                for action in valid_actions:
                    child = current.create_child(action, 0)
                    
                    tree_board = game.board.copy()
                    tree_board.step(action)
                    
                    # Rollout Policy
                    for n in range(1, 60):
                        rollout_board = tree_board.copy()
                        valid_rollout_actions = rollout_board.valid_moves()
                        if not valid_rollout_actions:
                            continue

                        while True:
                            rollout_action = np.random.choice(valid_rollout_actions)
                            _, _, done, valid_rollout_actions = rollout_board.step(rollout_action)
                            if done:
                                break
                                
                        # Backup
                        backup = child
                        while True:
                            backup.value += (1 / n) * (rollout_board.score - backup.value)
                            backup = backup.parent
                            if backup is None:
                                break

                # Real Policy - take only one step 
                current = root
                action = epsilon_greedy_policy(current, valid_actions, 0)
                _, _, done, valid_actions = game.board.step(action)
                
                redraw_window(win)

                if done:
                    print(f'MCTS Score: {game.board.score}')
                    mcts_scores.append(game.board.score)
                    break

                current = root.children[action]
                root = current
                root.parent = None

        print('Random:', np.mean(random_scores))
        print('MCTS:', np.mean(mcts_scores))