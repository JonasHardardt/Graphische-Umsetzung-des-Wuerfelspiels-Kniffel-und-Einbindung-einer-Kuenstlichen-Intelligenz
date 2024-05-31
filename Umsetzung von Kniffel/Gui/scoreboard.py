import numpy as np
class Scoreboard(object):
    def __init__(self):
        self.listeGemacht = np.zeros(13, dtype = int)
        self.listePunkte = np.zeros(13, dtype = int)
        self.summen = np.zeros(5, dtype = int)
    
    def eintrag(self, aktion, wuerfel):
        if self.listeGemacht[aktion] == 0:
            self.listeGemacht[aktion] = 1
            if aktion >= 1 and aktion <= 6:
                reward = wuerfel.count(aktion)*aktion
                self.summen[0] += reward
            else:
                gewuerfelteZahlen = []
                for i in range(1, 7):
                    gewuerfelteZahlen += [wuerfel.count(i)]
                if aktion == 7:
                    if 3 in gewuerfelteZahlen:
                        reward = wuerfel[0] + wuerfel[1] + wuerfel[2] + wuerfel[3] + wuerfel[4]
                    else:
                        reward = 0
                elif aktion == 8:
                    if 4 in gewuerfelteZahlen:
                        reward = wuerfel[0] + wuerfel[1] + wuerfel[2] + wuerfel[3] + wuerfel[4]
                    else:
                        reward = 0
                elif aktion == 9:
                    if 2 in gewuerfelteZahlen and 3 in gewuerfelteZahlen:
                        reward = 25
                    else:
                        reward = 0
                elif aktion == 10:
                    if (1 in self._wuerfel and 2 in self._wuerfel and 3 in self._wuerfel and 4 in self._wuerfel) or (2 in self._wuerfel and 3 in self._wuerfel and 4 in self._wuerfel and 5 in self._wuerfel) or (3 in self._wuerfel and 4 in self._wuerfel and 5 in self._wuerfel and 6 in self._wuerfel):
                        reward = 30
                    else:
                        reward = 0
                elif aktion == 11:
                    if (1 in self._wuerfel and 2 in self._wuerfel and 3 in self._wuerfel and 4 in self._wuerfel and 5 in self._wuerfel) or (2 in self._wuerfel and 3 in self._wuerfel and 4 in self._wuerfel and 5 in self._wuerfel and 6 in self._wuerfel):
                        reward = 40
                    else:
                        reward = 0
                elif aktion == 12:
                    if 5 in gewuerfelteZahlen:
                        reward = wuerfel[0] + wuerfel[1] + wuerfel[2] + wuerfel[3] + wuerfel[4]
                    else:
                        reward = 0
                elif aktion == 13:
                    reward = (wuerfel[0] + wuerfel[1] + wuerfel[2] + wuerfel[3] + wuerfel[4])/2
                self.summen[1] += reward
            self.listePunkte[aktion] = reward
            self.summen[2] = self.summen[0] + self.summen[1]
            if self.summen[0] >= 63:
                self.summen[2] += 35
        else:
            reward = -1
        
        
        return reward, self.listeGemacht, self.listePunkte, self.summen
s = Scoreboard()
s.eintrag(10, [1, 2, 3, 4, 5])
print(s.listeGemacht)
print(s.listePunkte)
print(s.summen)