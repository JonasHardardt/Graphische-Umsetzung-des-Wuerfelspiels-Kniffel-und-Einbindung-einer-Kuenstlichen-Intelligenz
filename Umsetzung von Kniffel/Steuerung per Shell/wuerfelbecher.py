from wuerfel import Wuerfel

class Wuerfelbecher(object):
    def __init__(self):
        self.wuerfel = []
        for i in range(5):
            self.wuerfel += [Wuerfel()]
            
    def wuerfelnAlle(self, ausgewaehlteWuerfel):
        ergebnisse = []
        zaehler = 0
        for i in self.wuerfel:
            if ausgewaehlteWuerfel[zaehler] == True:
                i.wuerfeln()
            ergebnisse += [i.getAugen()]
            zaehler += 1 
        return ergebnisse 
        
            
            