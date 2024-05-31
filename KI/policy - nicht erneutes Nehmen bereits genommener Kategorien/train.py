from KniffelInt import KniffelEnv
from PolicyGradient import PolicyNet
import numpy as np
import matplotlib.pyplot as plt


def trainAgent(numEpisodes):
    env = KniffelEnv()
    net = PolicyNet(18, 13, 18)
    steps = []
    #avgSteps = []
    collectRewards = []
    collectSums = []
    avgSum = []
    avgSteps = []
    avgRewards = []
    x1 = []
    for episode in range(1, numEpisodes + 1):
        gammaBeg = 0.2
        gammaEnd = 0.8
        state = env.reset()
        logProbs = []
        rewards = []
        step = 0
        done = False
        iteration = 1000
        x = []
        if episode % iteration == 0:
            x1 += [episode]
            avgSum += [np.mean(collectSums[-iteration:])]
            avgSteps += [np.mean(steps[-iteration:])]
            avgRewards += [np.mean(collectRewards[-iteration:])]
            print(episode, np.mean(collectRewards[-iteration:]), np.mean(collectSums[-iteration:]), np.mean(steps[-iteration:]))
        while not done:
            action, logProb = net.get_action(state)
            nextState, reward, done, sum = env.step(action)
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
    plt.savefig('3.1.png')
    plt.figure(2)
    plt.savefig('3.2.png')

    # Show the plots
    plt.show()



trainAgent(100000)