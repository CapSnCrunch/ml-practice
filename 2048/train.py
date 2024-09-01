from engine import Board
import matplotlib.pyplot as plt
from reinforce import PolicyApproximation, reinforce

if __name__ == '__main__':
    board = Board(4)
    pi = PolicyApproximation(4, 0.1)

    scores = reinforce(board=board, gamma=1, num_episodes=100, pi=pi)

    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.show()