from KniffelBesteAktion import KniffelEnv
from PolicyGradient import PolicyNet
import numpy as np
from berechnung import *
import torch
import matplotlib.pyplot as plt

def trainAgent(numEpisodes, nameNet):
    env1 = KniffelEnv()
    env2 = KniffelEnv()
    #net = torch.load(nameNet)
    net1 = PolicyNet(19, 13, 15)
    net2 = PolicyNet(19, 13, 15)
    steps1 = []
    steps2 = []
    collectRewards1 = []
    collectSums1 = []
    avgSum1 = []
    avgSteps1 = []
    avgRewards1 = []
    x1 = []
    collectRewards2 = []
    collectSums2 = []
    avgSum2 = []
    avgSteps2 = []
    avgRewards2 = []
    x2 = []
    x = []
    sum1 = 0
    sum2 = 0
    win = 100000
    for episode in range(1, numEpisodes + 1):
        #print()
        #print()
        gammaBeg = 0.5
        gammaEnd = 0.8
        state1 = env1.reset()
        state2 = env2.reset()
        #print(state)
        logProbs1 = []
        logProbs2 = []
        rewards1 = []
        rewards2 = []
        step1 = 0
        step2 = 0
        done1 = False
        done2 = False
        iteration = 1000
        if episode % iteration == 0:
            x1 += [episode]
            avgSum1 += [np.mean(collectSums1[-iteration:])]
            avgSteps1 += [np.mean(steps1[-iteration:])]
            avgRewards1 += [np.mean(collectRewards1[-iteration:])]
            x2 += [episode]
            avgSum2 += [np.mean(collectSums2[-iteration:])]
            avgSteps2 += [np.mean(steps2[-iteration:])]
            avgRewards2 += [np.mean(collectRewards2[-iteration:])]
            print(x1[-1], avgSum1[-1], avgSteps1[-1], avgRewards1[-1])
            print(x2[-1], avgSum2[-1], avgSteps2[-1], avgRewards2[-1])

        while not done1 and not done2:
            gleich = False
            while state1[0] <= 2 and not gleich:
                wuerfelNorminalisiert = state1[1:6]
                wuerfel = wuerfelWiederherstellen(wuerfelNorminalisiert)
                wuerfelBehalten = berechneWuerfelBehalten(wuerfel.tolist(), 2-state1[0], state1[6:].tolist())
                #print('behalten', wuerfelBehalten[1])
                state1 = env1.neuWuerfeln(wuerfelBehalten[1])
            if  not done1:
                action, logProb = net1.get_action(state1)
                nextState, reward, done1, sum1 = env1.greedy(action)
                logProbs1 += [logProb]
                rewards1 += [reward]
                step1 += 1
                state1 = nextState


            gleich = False
            while state2[0] <= 2 and not gleich:
                wuerfelNorminalisiert = state2[1:6]
                wuerfel = wuerfelWiederherstellen(wuerfelNorminalisiert)
                wuerfelBehalten = berechneWuerfelBehalten(wuerfel.tolist(), 2-state2[0], state2[6:].tolist())
                #print('behalten', wuerfelBehalten[1])
                state2 = env2.neuWuerfeln(wuerfelBehalten[1])
            if not done2:
                action, logProb = net2.get_action(state2)
                nextState, reward, done2, sum2 = env2.greedy(action)
                logProbs2 += [logProb]
                rewards2 += [reward]
                step2 += 1
                state2 = nextState
            #print(step1, step2)
        if episode > win:
            if sum1 > sum2:
                k = 1
            else:
                k = -1
            #print(rewards1, rewards2)
            for i in range(len(rewards1)):
                rewards1[i] = rewards1[i] + k*5
            for i in range(len(rewards2)):
                rewards2[i] = rewards2[i] + k*(-5)
            #print(rewards1, rewards2)
        gamma = round((gammaEnd - gammaBeg)* episode/numEpisodes + gammaBeg, 2)
        #print(gamma)
        net1.update_policy(rewards1, logProbs1, gamma)
        steps1 += [step1]
        collectRewards1 += [np.sum(rewards1)]
        collectSums1 += [sum1]

        net2.update_policy(rewards2, logProbs2, gamma)
        steps2 += [step2]
        collectRewards2 += [np.sum(rewards2)]
        collectSums2 += [sum2]
    #print(steps1)
    #print(collectRewards1)
    # Plot the data


    # Add legend


    plt.figure(1)
    plt.plot(x1, avgSum1, label='erreichte Summen')
    plt.plot(x1, avgSteps1, label='Anzahl der Runden')
    plt.title('Auswertung erreichte Summen/Runden')
    plt.xlabel('Anzahl Iterationen')
    plt.ylabel('Runden/Punkte')

    # Plot the second graph
    plt.figure(2)
    plt.plot(x1, avgRewards1)
    plt.title('Auswertung Rewards')
    plt.xlabel('Anzahl Iterationen')
    plt.ylabel('Punkte')
    plt.legend()

    plt.figure(3)
    plt.plot(x2, avgSum2, label='erreichte Summen')
    plt.plot(x2, avgSteps2, label='Anzahl der Runden')
    plt.title('Auswertung erreichte Summen/Runden')
    plt.xlabel('Anzahl Iterationen')
    plt.ylabel('Runden/Punkte')

    # Plot the second graph
    plt.figure(4)
    plt.plot(x2, avgRewards2)
    plt.title('Auswertung Rewards')
    plt.xlabel('Anzahl Iterationen')
    plt.ylabel('Punkte')
    plt.legend()

    # Save the plots as images
    plt.figure(1)
    plt.savefig('3.1.png')
    plt.figure(2)
    plt.savefig('3.2.png')
    plt.figure(3)
    plt.savefig('3.3.png')
    plt.figure(4)
    plt.savefig('3.4.png')

    # Show the plots
    plt.show()

    return net1


net = trainAgent(400000, 'policyNet1.pth')
#torch.save(net, 'policyNet1.pth')