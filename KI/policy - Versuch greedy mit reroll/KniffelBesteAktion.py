import numpy as np

class KniffelEnv(object):
    def __init__(self):
        self.state = []
        self.neugewuerfelt = 0
        self.wuerfel = np.random.randint(1, 6, size = 5)
        self.listeGemacht = np.zeros(13, dtype = int)
        self.listePunkte = np.zeros(13, dtype = int)
        self.summen = np.zeros(3, dtype = int)
        self.episode_ended = False
        self.current_time_step = None


    def getListeGemacht(self):
        return self.listeGemacht

    def normalisiereWuerfel(self, wuerfel):
        wuerfelN = np.array([0,0,0,0,0], dtype='float')
        for i in range(5):
            wuerfelN[i] = wuerfel[i]/10
        return wuerfelN

    def reset(self):
        self.wuerfel = np.random.randint(1, 6, size = 5)
        self.listeGemacht = np.zeros(13, dtype = int)
        self.listePunkte = np.zeros(13, dtype = int)
        self.summen = np.zeros(3, dtype = int)
        self.state = np.concatenate(([self.neugewuerfelt], self.normalisiereWuerfel(self.wuerfel), self.listeGemacht))
        self.episode_ended = False
        self.neugewuerfelt = 0
        self.current_time_step = dict(oberservation = [np.concatenate((self.normalisiereWuerfel(self.wuerfel), self.listeGemacht, self.summen))], dtype=np.int64)
        return self.state

    def getPunktzahl(self, aktion, gewuerfelteZahlen):
        if self.listeGemacht[aktion] == 0:
            if aktion >= 0 and aktion <= 5:
                punkte = np.count_nonzero(self.wuerfel == aktion + 1) * (aktion + 1)
            else:
                if aktion == 6:
                    if 3 in gewuerfelteZahlen or 4 in gewuerfelteZahlen or 5 in gewuerfelteZahlen:
                        punkte = self.wuerfel[0] + self.wuerfel[1] + self.wuerfel[2] + self.wuerfel[3] + self.wuerfel[4]
                    else:
                        punkte = 0
                elif aktion == 7:
                    if 4 in gewuerfelteZahlen or 5 in gewuerfelteZahlen:
                        punkte = self.wuerfel[0] + self.wuerfel[1] + self.wuerfel[2] + self.wuerfel[3] + self.wuerfel[4]
                    else:
                        punkte = 0
                elif aktion == 8:
                    if 2 in gewuerfelteZahlen and 3 in gewuerfelteZahlen:
                        punkte = 25
                    else:
                        punkte = 0
                elif aktion == 9:
                    if (1 in self.wuerfel and 2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel) or (2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel) or (3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel and 6 in self.wuerfel):
                        punkte = 30
                    else:
                        punkte = 0
                elif aktion == 10:
                    if (1 in self.wuerfel and 2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel) or (2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel and 6 in self.wuerfel):
                        punkte = 40
                    else:
                        punkte = 0
                elif aktion == 11:
                    if 5 in gewuerfelteZahlen:
                        punkte = 50
                    else:
                        punkte = 0
                elif aktion == 12:
                    punkte = (self.wuerfel[0] + self.wuerfel[1] + self.wuerfel[2] + self.wuerfel[3] + self.wuerfel[4])//2
        else:
            punkte = -10
        return punkte

    def getPunkteFuerJedeAktion(self):
        gewuerfelteZahlen = []
        for i in range(1, 7):
            gewuerfelteZahlen += [np.count_nonzero(self.wuerfel == i)]
        punktzahlen = []
        for aktion in range(13):
            punktzahlen += [self.getPunktzahl(aktion, gewuerfelteZahlen)]
        return punktzahlen

    def greedy(self, aktionNet):
        allePunktzahlen = self.getPunkteFuerJedeAktion()
        if aktionNet == allePunktzahlen.index(max(allePunktzahlen)):
            reward = max(allePunktzahlen)
        else:
            reward = (allePunktzahlen[aktionNet] - max(allePunktzahlen))*2
        nextState, pReward, done, sum = self.step(aktionNet)
        if pReward == -10:
            reward = -100
        return nextState, reward, done, sum


    def step(self, aktion):
        if self.listeGemacht[aktion] == 0:
            self.listeGemacht[aktion] = 1
            if aktion >= 0 and aktion <= 5:
                reward = np.count_nonzero(self.wuerfel == aktion + 1) * (aktion + 1)
                self.summen[0] += reward
            else:
                gewuerfelteZahlen = []
                for i in range(1, 7):
                    gewuerfelteZahlen += [np.count_nonzero(self.wuerfel == i)]
                if aktion == 6:
                    if 3 in gewuerfelteZahlen or 4 in gewuerfelteZahlen or 5 in gewuerfelteZahlen:
                        reward = self.wuerfel[0] + self.wuerfel[1] + self.wuerfel[2] + self.wuerfel[3] + self.wuerfel[4]
                    else:
                        reward = 0
                elif aktion == 7:
                    if 4 in gewuerfelteZahlen or 5 in gewuerfelteZahlen:
                        reward = self.wuerfel[0] + self.wuerfel[1] + self.wuerfel[2] + self.wuerfel[3] + self.wuerfel[4]
                    else:
                        reward = 0
                elif aktion == 8:
                    if 2 in gewuerfelteZahlen and 3 in gewuerfelteZahlen:
                        reward = 25
                    else:
                        reward = 0
                elif aktion == 9:
                    if (1 in self.wuerfel and 2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel) or (2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel) or (3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel and 6 in self.wuerfel):
                        reward = 30
                    else:
                        reward = 0
                elif aktion == 10:
                    if (1 in self.wuerfel and 2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel) or (2 in self.wuerfel and 3 in self.wuerfel and 4 in self.wuerfel and 5 in self.wuerfel and 6 in self.wuerfel):
                        reward = 40
                    else:
                        reward = 0
                elif aktion == 11:
                    if 5 in gewuerfelteZahlen:
                        reward = 50
                    else:
                        reward = 0
                elif aktion == 12:
                    reward = (self.wuerfel[0] + self.wuerfel[1] + self.wuerfel[2] + self.wuerfel[3] + self.wuerfel[4])/2
                self.summen[1] += reward
            self.listePunkte[aktion] = reward
            self.summen[2] = self.summen[0] + self.summen[1]
            if self.summen[0] >= 63:
                self.summen[2] += 35
        else:
            reward = -10
        if reward == 0:
            reward = -50
        self.wuerfel = np.random.randint(1, 6, size = 5)
        self.neugewuerfelt = 0
        self.state = np.concatenate(([self.neugewuerfelt], self.normalisiereWuerfel(self.wuerfel), self.listeGemacht))
        self.current_time_step = dict(state = np.concatenate((self.normalisiereWuerfel(self.wuerfel), self.listeGemacht, self.summen)), reward = reward, dtype=np.int64)
        self.done = False
        if np.count_nonzero(self.listeGemacht) == 13:
            self.done = True
        return self.state, reward, self.done, self.summen[-1]

    def neuWuerfeln(self, wuerfelBehalten):
        self.wuerfel = np.concatenate((np.random.randint(1, 6, size=5-len(wuerfelBehalten)), np.array(wuerfelBehalten)))
        self.neugewuerfelt += 1
        self.state = np.concatenate(([self.neugewuerfelt], self.normalisiereWuerfel(self.wuerfel), self.listeGemacht))
        return self.state