from spieler import Spieler

class Spielmanager(object):
    def __init__(self, pSpielerAnzahl, pNamen, pFenster, pBlock, pGui):
        self.spielerListe = []
        for name in pNamen:
            self.spielerListe += [Spieler(name, pFenster, pBlock, pGui)]
    
    def rundeDurchfuehren(self):
        for i in range(13):
            for spieler in self.spielerListe:
                print(f'Spieler {spieler.getName()}')
                spieler.spielen()
                print(spieler.getGesamt())
                print()
                