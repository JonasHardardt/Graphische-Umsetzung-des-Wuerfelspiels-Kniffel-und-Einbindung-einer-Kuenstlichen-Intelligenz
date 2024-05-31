from random import randint
from tkinter import *

class Wuerfel(object):
    
    def __init__(self, pFenster):
        self.augen = 0
        self.untereGrenze = 1
        self.obereGrenze = 6
        
    def wuerfeln(self):
        self.augen = randint(self.untereGrenze, self.obereGrenze)

        
        #return self.augen
        
    def getAugen(self):
        return self.augen