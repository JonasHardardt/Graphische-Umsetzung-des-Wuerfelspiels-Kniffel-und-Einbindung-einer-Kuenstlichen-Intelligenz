class Ereignismanager(object):
    def __init__(self):
        self.ereignisListe = ["1er", "2er", "3er", "4er", "5er", "6er", "dreierpasch", "viererpasch", "fullhouse", "kleinestrasse", "grossestrasse", "kniffel", "chance"]
        
    def getEreignisListe(self):
        return self.ereignisListe
    
    def eingabeBearbeiten(self, ereignis):
        ereignis = ereignis.lower()
        ereignis = ereignis.replace("ß", "ss")
        ereignis = ereignis.replace(" ", "")
        ereignis = ereignis.replace("streichen", "")
        if ereignis == 'einser':
            ereignis = ereignis.replace("einser", "1er")
        if ereignis == 'zweier':
            ereignis = ereignis.replace("zweier", "2er")
        if ereignis == 'dreier':
            ereignis = ereignis.replace("dreier", "3er")
        if ereignis == 'vierer':
            ereignis = ereignis.replace("vierer", "4er")
        if ereignis == 'fünfer':
            ereignis = ereignis.replace("fünfer", "5er")
        if ereignis == 'sechser':
            ereignis = ereignis.replace("sechser", "6er")
        if ereignis == '1':
            ereignis = ereignis.replace("1", "1er")
        if ereignis == '2':
            ereignis = ereignis.replace("2", "2er")
        if ereignis == '3':
            ereignis = ereignis.replace("3", "3er")
        if ereignis == '4':
            ereignis = ereignis.replace("4", "4er")
        if ereignis == '5':
            ereignis = ereignis.replace("5", "5er")
        if ereignis == '6':
            ereignis = ereignis.replace("6", "6er")
        return ereignis
    
    def ereignisHinzufuegen(self, pWuerfelErgebnis):
        ereignis = input("Welches Ereignis möchtest du ausüben?(z.B. Viererpasch, Große Strasse) " )
        if 'streichen' in ereignis:
            streichen = True
        else:
            streichen = False
        ereignis = self.eingabeBearbeiten(ereignis)
        while not ereignis in self.ereignisListe and not 'streichen' in ereignis:
            print(f"Das Ereignis {ereignis} existiert nicht oder du hast es bereits gespielt.")
            ereignis = input("Welches Ereignis möchtest du ausüben?(z.B. Viererpasch, Große Strasse) " )
            ereignis = self.eingabeBearbeiten(ereignis)
        else:
            gewuerfelteZahlen = []
            for i in range(1, 7):
                gewuerfelteZahlen += [pWuerfelErgebnis.count(i)]
            if ereignis == "1er":
                wert = gewuerfelteZahlen[0]*1
                if streichen == True:
                    wert = 0
                self.ereignisListe[0] = wert
            elif ereignis == "2er":
                wert = gewuerfelteZahlen[1]*2
                if streichen == True:
                    wert = 0
                self.ereignisListe[1] = wert
            elif ereignis == "3er":
                wert = gewuerfelteZahlen[2]*3
                if streichen == True:
                    wert = 0
                self.ereignisListe[2] = wert
            elif ereignis == "4er":
                wert = gewuerfelteZahlen[3]*4
                if streichen == True:
                    wert = 0
                self.ereignisListe[3] = wert
            elif ereignis == "5er":
                wert = gewuerfelteZahlen[4]*5
                if streichen == True:
                    wert = 0
                self.ereignisListe[4] = wert
            elif ereignis == "6er":
                wert = gewuerfelteZahlen[5]*6
                if streichen == True:
                    wert = 0
                self.ereignisListe[5] = wert
            elif ereignis == "dreierpasch":
                if streichen == True or 3 in gewuerfelteZahlen:
                    wert = pWuerfelErgebnis[0] + pWuerfelErgebnis[1] + pWuerfelErgebnis[2] + pWuerfelErgebnis[3] + pWuerfelErgebnis[4]
                    if streichen == True:
                        wert = 0
                    self.ereignisListe[6] = wert
                else:
                    self.ereignisHinzufuegen(self, pWuerfelErgebnis)
            elif ereignis == "viererpasch":
                if streichen == True or 4 in gewuerfelteZahlen:
                    wert = pWuerfelErgebnis[0] + pWuerfelErgebnis[1] + pWuerfelErgebnis[2] + pWuerfelErgebnis[3] + pWuerfelErgebnis[4]
                    if streichen == True:
                        wert = 0
                    self.ereignisListe[7] = wert
                else:
                    self.ereignisHinzufuegen(self, pWuerfelErgebnis)
            elif ereignis == "fullhouse":
                if streichen == True or 2 in gewuerfelteZahlen and 3 in gewuerfelteZahlen:
                    wert = 25
                    if streichen == True:
                        wert = 0
                    self.ereignisListe[8] = wert
                else:
                    print("Du kannst mit deinen Würfeln nicht Full House spielen, wähle ein anderes Ereignis.")
                    self.ereignisHinzufuegen(pWuerfelErgebnis)
            elif ereignis == "kleinestrasse":
                if streichen == True or (1 in pWuerfelErgebnis and 2 in pWuerfelErgebnis and 3 in pWuerfelErgebnis and 4 in pWuerfelErgebnis) or (5 in pWuerfelErgebnis and 2 in pWuerfelErgebnis and 3 in pWuerfelErgebnis and 4 in pWuerfelErgebnis) or (5 in pWuerfelErgebnis and 6 in pWuerfelErgebnis and 3 in pWuerfelErgebnis and 4 in pWuerfelErgebnis):
                    wert = 30
                    if streichen == True:
                        wert = 0
                    self.ereignisListe[9] = wert
                else:
                    print("Du kannst mit deinen Würfeln nicht die Kleine Straße spielen, wähle ein anderes Ereignis.")
                    self.ereignisHinzufuegen(pWuerfelErgebnis)
            elif ereignis == "grossestrasse":
                if streichen == True or (1 in pWuerfelErgebnis and 2 in pWuerfelErgebnis and 3 in pWuerfelErgebnis and 4 in pWuerfelErgebnis and 5 in pWuerfelErgebnis) or (5 in pWuerfelErgebnis and 2 in pWuerfelErgebnis and 3 in pWuerfelErgebnis and 4 in pWuerfelErgebnis and 6 in pWuerfelErgebnis):
                    wert = 40
                    if streichen == True:
                        wert = 0
                    self.ereignisListe[10] = wert
                else:
                    print("Du kannst mit deinen Würfeln nicht die Große Straße spielen, wähle ein anderes Ereignis.")
                    self.ereignisHinzufuegen(pWuerfelErgebnis)
            elif ereignis == "kniffel":
                if 5 in gewuerfelteZahlen or streichen == True:
                    wert = 50
                    if streichen == True:
                        wert = 0
                    self.ereignisListe[11] = wert
                else:
                    print("Du kannst mit deinen Würfeln nicht Kniffel spielen, wähle ein anderes Ereignis.")
                    self.ereignisHinzufuegen(pWuerfelErgebnis)
            elif ereignis == "chance":
                wert = pWuerfelErgebnis[0] + pWuerfelErgebnis[1] + pWuerfelErgebnis[2] + pWuerfelErgebnis[3] + pWuerfelErgebnis[4]
                if streichen == True:
                    wert = 0
                self.ereignisListe[12] = wert