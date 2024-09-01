import torch
import numpy as np
from torch import nn
from engine import Board
import torch.nn.functional as F

class PolicyApproximation(nn.Module):
    def __init__(self, num_actions, alpha):
        super().__init__()

        self.num_channels = 11
        self.num_actions = num_actions

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel, 32 output channels
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 64 output channels
            nn.ReLU()
        ).double()

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),  # Flattened size after convolutions
            nn.ReLU(),
            nn.Linear(64, num_actions)
        ).double()

        self.alpha = alpha
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def preprocess_state(self, state): 
        one_hot_state = np.zeros((4, 4, self.num_channels))  # Initialize a (17, 4, 4) array

        # Reshape the state back to 4x4 if it's flattened
        state = state.reshape(4, 4)

        for i in range(4):
            for j in range(4):
                if state[i, j] > 0:
                    one_hot_state[i, j, int(np.log2(state[i, j]))] = 1

        return one_hot_state

    def forward(self, x):
        x = x.view(-1, self.num_channels, 4, 4)  # Reshape input to 1x1x4x4 (batch_size, channels, height, width)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc_layers(x)
        x = F.softmax(x, dim=1)
        return x.squeeze()

    def __call__(self, s, valid_moves = None):
        if isinstance(s, np.ndarray):
            s = self.preprocess_state(s)
            s = torch.tensor(s, dtype=torch.float64)
        
        if valid_moves is None:
            valid_moves = torch.arange(self.num_actions)
        
        self.eval()
        with torch.no_grad():
            output = self.forward(s)

            # Mask invalid moves
            mask = torch.zeros_like(output)
            mask[valid_moves] = 1.0
            masked_output = output * mask
            
            # Renormalize to sum to 1
            if masked_output.sum() > 0:
                masked_output /= masked_output.sum()
                return torch.distributions.categorical.Categorical(masked_output).sample().item()
            else:
                return torch.distributions.categorical.Categorical(output).sample().item()
        
    def update(self, s, a, gamma_t, delta):
        if isinstance(s, np.ndarray):
            s = self.preprocess_state(s)
            s = torch.tensor(s, dtype=torch.float64)

        self.train()
        self.optimizer.zero_grad()

        output = self.forward(s)
        action_prob = -output[a].log()

        loss = self.alpha * gamma_t * delta * action_prob
        # print(f'loss {loss}, output {output}, action_prob {action_prob}')

        loss.backward()
        self.optimizer.step()

def reinforce(
    board:Board,
    gamma:float,
    num_episodes:int,
    pi:PolicyApproximation
):
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """

    scores = []
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f'Episode {episode}')

        s = np.array(board.restart()).reshape(-1)
        states = [s]
        actions = [pi(states[0])]
        rewards = [0]

        while True:
            Sp, R, done, valid_moves = board.step(actions[-1])
            
            Sp = np.array(Sp).reshape(-1)
            valid_moves = np.array(valid_moves)

            Ap = pi(Sp, valid_moves)
            # print(f'  Action {Ap}   Valid Next Moves ({valid_moves})')

            states.append(Sp)
            rewards.append(R)
            actions.append(Ap)

            if done:
                break

        G = 0
        T = len(states)
        for i in range(T):
            t = T - i - 1
            S = states[t]
            A = actions[t]
            R = rewards[t]

            G = R + gamma * G
            
            delta = G
            pi.update(S, A, gamma ** (T - 1 - t), delta)

        scores.append(sum(rewards))
        
    return scores