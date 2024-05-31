from KniffelBesteAktion import KniffelEnv
from PolicyGradient import PolicyNet
import numpy as np
from berechnung import *
import torch
import matplotlib.pyplot as plt

def trainAgent(numEpisodes, nameNet):
    global avgSum, avgSteps, avgRewards, x1
    env = KniffelEnv()
    #net = torch.load(nameNet)
    net = PolicyNet(19, 13, 15)
    steps = []
    collectRewards = []
    collectSums = []
    avgSum = []
    avgSteps = []
    avgRewards = []
    x = []
    x1 = []
    for episode in range(1, numEpisodes + 1):
        #print()
        #print()
        gammaBeg = 0.5
        gammaEnd = 0.8
        state = env.reset()
        #print(state)
        logProbs = []
        rewards = []
        step = 0
        done = False
        iteration = 1000
        if episode % iteration == 0:
            x1 += [episode]
            avgSum += [np.mean(collectSums[-iteration:])]
            avgSteps += [np.mean(steps[-iteration:])]
            avgRewards += [np.mean(collectRewards[-iteration:])]
            print(x1[-1], avgSum[-1], avgSteps[-1], avgRewards[-1])
        while not done:
            gleich = False
            while state[0] <= 2 and not gleich:
                wuerfelNorminalisiert = state[1:6]
                wuerfel = wuerfelWiederherstellen(wuerfelNorminalisiert)
                wuerfelBehalten = berechneWuerfelBehalten(wuerfel.tolist(), 2-state[0], state[6:].tolist())
                #print('behalten', wuerfelBehalten[1])
                state = env.neuWuerfeln(wuerfelBehalten[1])
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
        x += [episode]
    print(steps)
    print(collectRewards)
    # Plot the data


    # Add legend


    plt.figure(1)
    plt.plot(x1, avgSum, label='erreichte Summen')
    plt.plot(x1, avgSteps, label='Anzahl der Runden')
    plt.title('Auswertung erreichte Summen/Runden')
    plt.xlabel('Anzahl Iterationen')
    plt.ylabel('Runden/Punkte')

    # Plot the second graph
    plt.figure(2)
    plt.plot(x1, avgRewards)
    plt.title('Auswertung Rewards')
    plt.xlabel('Anzahl Iterationen')
    plt.ylabel('Punkte')
    plt.legend()

    # Save the plots as images
    plt.figure(1)
    plt.savefig('4.1.png')
    plt.figure(2)
    plt.savefig('4.2.png')

    # Show the plots
    plt.show()

    return net


net = trainAgent(10000, 'policyNet1.pth')
#torch.save(net, 'policyNet1.pth')