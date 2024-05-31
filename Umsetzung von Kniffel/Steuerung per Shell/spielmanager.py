from spieler import Spieler

class Spielmanager(object):
    def __init__(self, pSpielerAnzahl):
        self.spielerListe = []
        for i in range(pSpielerAnzahl):
            name = input(f'Wie soll der {i+1}. Spieler heißen? ')
            self.spielerListe += [Spieler(name)]
        self.rundeDurchfuehren()
    
    def rundeDurchfuehren(self):
        for i in range(13):
            for spieler in self.spielerListe:
                print(f'Spieler {spieler.getName()}')
                spieler.spielen()
                print(spieler.getGesamt())
                print()
                