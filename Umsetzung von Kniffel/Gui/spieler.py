from ereignismanager import Ereignismanager
from wuerfelbecher import Wuerfelbecher
from tkinter import *
from time import sleep

class Spieler(object):
    def __init__(self, pName):
        self.ereignisse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.gemacht = [False]*13
        self.name = pName
        self.summeObererTeil = 0
        self.summeUntererTeil = 0
        self.gesamt = 0
        self.wuerfel = []
        self.runde = 0
        self.bonus = 0
        
    def getName(self):
        return self.name

    def setWuerfel(self, pWuerfel):
        self.wuerfel = pWuerfel

    def inRunde(self):
        self.runde += 1

    def getRunde(self):
        return self.runde
    
    def setEreignis(self, ereignis):
        gewuerfelteZahlen = []
        for i in range(1, 7):
            gewuerfelteZahlen += [self.wuerfel.count(i)]
        self.ereignisse[ereignis] = self.getPunktzahl(ereignis, gewuerfelteZahlen)
        self.gemacht[ereignis] = True
        self.runde = 0
        self.summeObererTeil = sum(self.ereignisse[:6])
        self.summeUntererTeil = sum(self.ereignisse[6:])
        if self.summeObererTeil > 62:
            self.bonus = 35
        self.gesamt = self.summeObererTeil + self.summeUntererTeil + self.bonus

    def getSummen(self):
        return self.summeObererTeil, self.bonus, self.summeUntererTeil, self.gesamt


    def getEreignisse(self):
        return self.ereignisse
    
    def getAllePunktzahlen(self):
        gewuerfelteZahlen = []
        for i in range(1, 7):
            gewuerfelteZahlen += [self.wuerfel.count(i)]
        punktzahlen = []
        for aktion in range(13):
            punktzahlen += [self.getPunktzahl(aktion, gewuerfelteZahlen)]
        return punktzahlen

    def getGemacht(self):
        return self.gemacht

        
    def getPunktzahl(self, aktion, gewuerfelteZahlen):
        if aktion >= 0 and aktion <= 5:
            punkte = gewuerfelteZahlen[aktion] * (aktion + 1)
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
                punkte = (self.wuerfel[0] + self.wuerfel[1] + self.wuerfel[2] + self.wuerfel[3] + self.wuerfel[4])
        return punkte        
        
    
            