import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Constants
GAMMA = 0.9

class PolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=6e-4):
        super().__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=0)
        return x

    def get_action(self, state):
        state = torch.Tensor(np.array(state, dtype = float))
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, rewards, log_probs, gamma):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

