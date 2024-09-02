import numpy as np
from engine import Board
import concurrent.futures

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
    
def rollout_policy(tree_board):
    rollout_board = tree_board.copy()
    valid_rollout_actions = rollout_board.valid_moves()
    if not valid_rollout_actions:
        return

    while True:
        rollout_action = np.random.choice(valid_rollout_actions)
        _, _, done, valid_rollout_actions = rollout_board.step(rollout_action)
        if done:
            break
    
    return rollout_board.score

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

    # Monte Carlo Tree Search
    mcts_scores = []
    for i in range(100):
        
        alpha = 0.5
        root = Node()
        
        state, valid_actions = b.restart()
        
        count = 0
        while True:

            for epsilon in [1, 0.5, 0.25, 0.01]:

                current = root
                
                tree_board = b.copy()
                valid_tree_actions = tree_board.valid_moves()

                # Tree Policy
                while True:
                    tree_action = epsilon_greedy_policy(current, valid_tree_actions, epsilon)
                    _, _, done, valid_tree_actions = tree_board.step(tree_action)

                    if tree_action not in current.children.keys():
                        current.create_child(tree_action, current.value + 4)
                        current = current.children[tree_action]
                        break

                    current = current.children[tree_action]

                    if done:
                        break
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(rollout_policy, tree_board) for _ in range(30)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]

                    # Backup
                    for result in results:
                        backup = current
                        while True:
                            backup.value += alpha * (result - backup.value)
                            backup = backup.parent
                            if backup is None:
                                break

            # Real Policy - take only one step 
            current = root
            action = epsilon_greedy_policy(current, valid_actions, 0)
            _, _, done, valid_actions = b.step(action)
            
            # redraw_window(win)

            count += 1
            print(f'Move {count}...')

            if done:
                print(f'MCTS Score: {b.score}')
                mcts_scores.append(b.score)
                break

            current = root.children[action]
            root = current
            root.parent = None

        print('Random:', np.mean(random_scores))
        print('MCTS:', np.mean(mcts_scores))