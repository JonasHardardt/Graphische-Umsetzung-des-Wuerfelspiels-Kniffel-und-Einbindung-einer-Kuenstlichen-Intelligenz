from ereignismanager import Ereignismanager
from wuerfelbecher import Wuerfelbecher 

class Spieler(object):
    def __init__(self, pName):
        self.ereignismanager = Ereignismanager()
        self.name = pName
        self.summeObererTeil = 0
        self.summeUntererTeil = 0
        self.gesamt = 0
        self.wuerfelAlle = Wuerfelbecher()
        
        
    def spielen(self):
        runde = 1
        ausgewaehlteWuerfel = [True, True, True, True, True]
        while runde < 4:
            wuerfelErgebnis = self.wuerfelAlle.wuerfelnAlle(ausgewaehlteWuerfel)
            print(wuerfelErgebnis, 'hast du gewürfelt')
            if runde != 3:
                if input('Möchtest du nochmal würfeln? ').lower() == 'nein':
                    break
                wuerfelAuswaehlen =  input('Welche Würfel möchtest du behalten? ')
                for i in range(5):
                    if str(i+1) in wuerfelAuswaehlen:
                        ausgewaehlteWuerfel[i] = False
                    else:
                        ausgewaehlteWuerfel[i] = True
            runde += 1
        self.ereignismanager.ereignisHinzufuegen(wuerfelErgebnis)
            
        
    def setSummen(self):
        summeObenListe = self.ereignismanager.getEreignisListe()[0:5]
        summeOben = 0
        for i in summeObenListe:
            if type(i) == int:
                summeOben += i
        if summeOben >= 63:
            summeOben += 35
        self.summeObererTeil = summeOben
        summeUntenListe = self.ereignismanager.getEreignisListe()[6:]
        summeUnten = 0
        for i in summeUntenListe:
            if type(i) == int:
                summeUnten += i
        self.summeUntererTeil = summeUnten
        self.gesamt = self.summeObererTeil + self.summeUntererTeil
        
    def getName(self):
        return self.name
    
    def getGesamt(self):
        self.setSummen()
        return self.gesamt
        