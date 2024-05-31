import sys
import torch
#import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt



class PolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-3):
        super().__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        '''x = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x))
        x2 = F.relu(self.linear3(x1))
        x3 = F.relu(self.linear4(x2))
        x4 = F.relu(self.linear5(x3))
        x5 = F.relu(self.linear6(x4))
        x6 = F.relu(self.linear7(x5))
        x7 = F.relu(self.linear8(x6))
        x8 = F.relu(self.linear9(x7))
        x9 = F.relu(self.linear10(x8))
        x10 = F.relu(self.linear11(x9))
        x11 = F.relu(self.linear12(x10))
        x12 = F.softmax(self.linear13(x11), dim=0)
        #print(x)'''
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=0)
        return x

    def get_action(self, state):
        #print(state)
        state = torch.Tensor(np.array(state, dtype = float))
        #print(state)
        probs = self.forward(Variable(state))
        #p=np.squeeze(probs.detach().numpy())
        #highest_prob_action = np.where(p == max(p))[0][0]
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        #print(probs)
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
        #print(discounted_rewards)

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()


#p1 = PolicyNetwork(4, 2, 3)
#action, problog = p1.get_action([1, 2, 3, 4])
#action2, problog2 = p1.get_action([1, 2, 2, 2])
#update_policy(p1, [5, -5], [problog, problog2])
#print(p1.get_action([1, 2, 3, 4]))
#print(p1.get_action([1, 2, 2, 2]))

