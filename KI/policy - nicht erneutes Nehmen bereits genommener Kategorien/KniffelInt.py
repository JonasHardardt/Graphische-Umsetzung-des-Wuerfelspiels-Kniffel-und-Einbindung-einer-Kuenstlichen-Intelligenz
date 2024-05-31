import numpy as np

class KniffelEnv(object):
    def __init__(self):
        self._action_spec = dict(
            dtype=np.int64, minimum=0, maximum=12, name='action')
        self._observation_spec = dict(shape=[21,1], dtype=np.int64, minimum=0, name='observation')
        self.state = []
        self.wuerfel = np.random.randint(1, 6, size = 5)
        self.listeGemacht = np.zeros(13, dtype = int)
        self.listePunkte = np.zeros(13, dtype = int)
        self.summen = np.zeros(3, dtype = int)
        self.episode_ended = False
        self.current_time_step = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def getListeGemacht(self):
        return self.listeGemacht

    def normalisiereWuerfel(self, wuerfel):
        wuerfelN = np.array([0,0,0,0,0], dtype='float')
        for i in range(5):
            wuerfelN[i] = wuerfel[i]/10
        #print(wuerfelN)
        return wuerfelN

    def reset(self):
        self.wuerfel = np.random.randint(1, 6, size = 5)
        self.listeGemacht = np.zeros(13, dtype = int)
        self.listePunkte = np.zeros(13, dtype = int)
        self.summen = np.zeros(3, dtype = int)
        self.state = np.concatenate((self.normalisiereWuerfel(self.wuerfel), self.listeGemacht))
        self.episode_ended = False
        self.current_time_step = dict(oberservation = [np.concatenate((self.normalisiereWuerfel(self.wuerfel), self.listeGemacht, self.summen))], dtype=np.int64)
        return self.state

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
            reward = -5
        self.wuerfel = np.random.randint(1, 6, size = 5)
        self.state = np.concatenate((self.normalisiereWuerfel(self.wuerfel), self.listeGemacht))
        self.current_time_step = dict(state = np.concatenate((self.normalisiereWuerfel(self.wuerfel), self.listeGemacht, self.summen)), reward = reward, dtype=np.int64)
        self.done = False
        if np.count_nonzero(self.listeGemacht) == 13:
            self.done = True
        return self.state, reward, self.done, self.summen[-1]