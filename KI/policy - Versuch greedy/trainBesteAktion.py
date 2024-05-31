from KniffelBesteAktion import KniffelEnv
from PolicyGradient import PolicyNet
import numpy as np
import torch

def trainAgent(numEpisodes, nameNet):
    env = KniffelEnv()
    #net = torch.load(nameNet)
    net = PolicyNet(18, 13, 15)
    steps = []
    #avgSteps = []
    collectRewards = []
    collectSums = []

    for episode in range(1, numEpisodes + 1):
        #print()
        #print()
        gammaBeg = 0.2
        gammaEnd = 0.8
        state = env.reset()
        #print(state)
        logProbs = []
        rewards = []
        step = 0
        done = False
        iteration = 500
        if episode % iteration == 0:
            print(episode, np.mean(collectRewards[-iteration:]), np.mean(collectSums[-iteration:]), np.mean(steps[-iteration:]))
        while not done:
            action, logProb = net.get_action(state)
            #print(action)
            nextState, reward, done, sum = env.greedy(action)
            #print(nextState, reward, done, sum)
            #print(reward)
            logProbs += [logProb]
            rewards += [reward]
            step += 1
            state = nextState
        gamma = round((gammaEnd - gammaBeg)* episode/numEpisodes + gammaBeg, 2)
        #print(gamma)
        net.update_policy(rewards, logProbs, gamma)
        steps += [step]
        collectRewards += [np.sum(rewards)]
        collectSums += [sum]
    print(steps)
    print(collectRewards)
    return net


net = trainAgent(200000, 'policyNet1.pth')
#torch.save(net, 'policyNet1.pth')