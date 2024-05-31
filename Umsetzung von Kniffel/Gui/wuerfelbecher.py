from wuerfel import Wuerfel
from tkinter import *

class Wuerfelbecher(object):
    def __init__(self, pFenster, pGui):
        self.wuerfel = []
        self.fenster = pFenster
        self.gui = pGui
        for i in range(5):
            self.wuerfel += [Wuerfel(pFenster)]
            
    def wuerfelnAlle(self, ausgewaehlteWuerfel):
        ergebnisse = []
        zaehler = 0
        for i in self.wuerfel:
            if ausgewaehlteWuerfel[zaehler] == True:
                i.wuerfeln()
            ergebnisse += [i.getAugen()]
            zaehler += 1
        return ergebnisse 
        
            
            