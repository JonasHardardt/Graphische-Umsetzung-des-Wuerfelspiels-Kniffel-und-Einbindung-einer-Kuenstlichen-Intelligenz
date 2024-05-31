from tkinter import *
from random import randint
from spieler import Spieler

class guimanager(object):
    def __init__(self):
        self.anzahlSpieler = 2
        self.spielmanager = None
        self.fenster = None
        self.block = None
        self.namen = ['1', '2']
        self.labelWuerfel = []
        self.labelWuerfel1 = None
        self.labelWuerfel2 = None
        self.labelWuerfel3 = None
        self.labelWuerfel4 = None
        self.labelWuerfel5 = None
        self.aktuellerSpieler = None
        self.spieler = []
        self.spielerLabel = []
        self.kategorien = []
        self.spielen()
    
    def starteSpiel(self):
        tkFenster = Tk()
        tkFenster.title('Start')
        # Aktivierung des Fensters
        tkFenster.geometry('300x200')
        label = Label(master=tkFenster, text='Wie viele Spieler (2-4) sollen mitspielen?', compound = CENTER)
        label.place(x=10, y=5)
        def buttonWeiterClick():
            if eingabeSpieler.get() != '':
                self.anzahlSpieler = int(eingabeSpieler.get())
            else:
                self.anzahlSpieler = 0
            tkFenster.quit()
            tkFenster.destroy()
            if self.anzahlSpieler > 4 or self.anzahlSpieler < 2:
                self.starteSpiel()
            else:
                self.abfrageNamen()
        buttonWeiter = Button(master=tkFenster, text='weiter', command=buttonWeiterClick)
        buttonWeiter.place(x=125, y=130, width=50, height=30)
        eingabeSpieler = Entry(master=tkFenster, bg='white')
        eingabeSpieler.place(x=125, y=64, width=50, height=27)
        tkFenster.mainloop()

    def abfrageNamen(self):
        for i in range(self.anzahlSpieler):
            tkFenster = Tk()
            tkFenster.title('Name Spieler')
            # Aktivierung des Fensters
            tkFenster.geometry('300x200')
            text1 = f'Wie soll der {i+1}. Spieler heißen?'
            label = Label(master=tkFenster, text=text1, compound = CENTER)
            label.place(x=10, y=5)
            def buttonWeiterClick():
                self.namen += [eingabeSpieler.get()]
                tkFenster.quit()
                tkFenster.destroy()
            buttonWeiter = Button(master=tkFenster, text='weiter', command=buttonWeiterClick)
            buttonWeiter.place(x=125, y=130, width=50, height=30)
            eingabeSpieler = Entry(master=tkFenster, bg='white')
            eingabeSpieler.place(x=125, y=64, width=50, height=27)
            tkFenster.mainloop()
        self.spielmanager = None
        #Spielmanager(self.anzahlSpieler, self.namen, self.fenster, self.block, self)
        
        self.spielen()
    
    def spielen(self):
        for name in self.namen:
            self.spieler += [Spieler(name)]
        self.aktuellerSpieler = self.spieler[0]
        self.fenster = Tk()
        self.fenster.title('Spiel')
        self.fenster.geometry('1100x900+0+0')
        self.block = Tk()
        self.block.title('Kniffelblock')
        self.block.geometry('600x900+1250+0')
        fontsize = 15
        label1er = Label(master=self.block, text='1er', font=("Arial", fontsize))
        label1er.grid(row=1, column=0, padx='5', pady='5', sticky='w')
        label2er = Label(master=self.block, text='2er', font=("Arial", fontsize))
        label2er.grid(row=2, column=0, padx='5', pady='5', sticky='w')
        label3er = Label(master=self.block, text='3er', font=("Arial", fontsize))
        label3er.grid(row=3, column=0, padx='5', pady='5', sticky='w')
        label4er = Label(master=self.block, text='4er', font=("Arial", fontsize))
        label4er.grid(row=4, column=0, padx='5', pady='5', sticky='w')
        label5er = Label(master=self.block, text='5er', font=("Arial", fontsize))
        label5er.grid(row=5, column=0, padx='5', pady='5', sticky='w')
        label6er = Label(master=self.block, text='6er', font=("Arial", fontsize))
        label6er.grid(row=6, column=0, padx='5', pady='5', sticky='w')
        labelBonus = Label(master=self.block, text='Bonus', font=("Arial", fontsize))
        labelBonus.grid(row=15, column=0, padx='5', pady='10', sticky='w')
        labelSummeOben = Label(master=self.block, text='Summe oben', font=("Arial", fontsize))
        labelSummeOben.grid(row=14, column=0, padx='5', pady='5', sticky='w')
        labelDreierpasch = Label(master=self.block, text='Dreierpasch', font=("Arial", fontsize))
        labelDreierpasch.grid(row=7, column=0, padx='5', pady='10', sticky='w')
        labelViererpasch = Label(master=self.block, text='Viererpasch', font=("Arial", fontsize))
        labelViererpasch.grid(row=8, column=0, padx='5', pady='5', sticky='w')
        labelFullHouse = Label(master=self.block, text='Full House', font=("Arial", fontsize))
        labelFullHouse.grid(row=9, column=0, padx='5', pady='5', sticky='w')
        labelKleineStrasse = Label(master=self.block, text='Kleine Straße', font=("Arial", fontsize))
        labelKleineStrasse.grid(row=10, column=0, padx='5', pady='5', sticky='w')
        labelGrosseStrasse = Label(master=self.block, text='Große Straße', font=("Arial", fontsize))
        labelGrosseStrasse.grid(row=11, column=0, padx='5', pady='5', sticky='w')
        labelKniffel = Label(master=self.block, text='Kniffel', font=("Arial", fontsize))
        labelKniffel.grid(row=12, column=0, padx='5', pady='5', sticky='w')
        labelChance = Label(master=self.block, text='Chance', font=("Arial", fontsize))
        labelChance.grid(row=13, column=0, padx='5', pady='5', sticky='w')
        labelSummeUnten = Label(master=self.block, text='Summe unten', font=("Arial", fontsize))
        labelSummeUnten.grid(row=16, column=0, padx='5', pady='10', sticky='w')
        labelSummeGesamt = Label(master=self.block, text='Summe gesamt', font=("Arial", fontsize))
        labelSummeGesamt.grid(row=17, column=0, padx='5', pady='10', sticky='w')

        def checkWuerfel(wuerfel):
            if self.aktuellerSpieler.getRunde() == 3:
                button["state"] = DISABLED
            else:
                button["state"] = NORMAL
            self.aktuellerSpieler.setWuerfel(wuerfel)
            self.labelRunde.config(text=str(self.aktuellerSpieler.getRunde()))
            self.labelNameSpieler.config(text=str(self.aktuellerSpieler.getName()))

        def checkSpielstandVorschau():
            index = self.spieler.index(self.aktuellerSpieler)
            punktzahlen = self.aktuellerSpieler.getAllePunktzahlen()
            for i in range(13):
                self.spielerLabel[index][i].config(text=str(punktzahlen[i]))
                self.spielerLabel[index][i].config(fg='gray')
        
        def buttonWuerfelnClick():
            if not var1.get():
                self.labelWuerfel1.config(text = str(randint(1, 6)))
            if not var2.get():
                self.labelWuerfel2.config(text = str(randint(1, 6)))
            if not var3.get():
                self.labelWuerfel3.config(text = str(randint(1, 6)))
            if not var4.get():
                self.labelWuerfel4.config(text = str(randint(1, 6)))
            if not var5.get():
                self.labelWuerfel5.config(text = str(randint(1, 6)))
            self.aktuellerSpieler.inRunde()
            self.labelRunde.config(text=str(self.aktuellerSpieler.getRunde()))
            w1 = self.labelWuerfel1.cget('text')
            w2 = self.labelWuerfel2.cget('text')
            w3 = self.labelWuerfel3.cget('text')
            w4 = self.labelWuerfel4.cget('text')
            w5 = self.labelWuerfel5.cget('text')
            wuerfel = [int(w1), int(w2), int(w3), int(w4), int(w5)]
            checkWuerfel(wuerfel)
            checkSpielstand()
            checkSpielstandVorschau()
            aktualisiereWuerfel()

        def setSpielerLabel():
            weite = 10
            namen = []
            for i in range(self.anzahlSpieler):
                namen += [Label(master = self.block, text = self.namen[i], width = weite)]
                namen[-1].grid(row=0, column=i+1, padx='10', pady='5', sticky='w')
                self.spielerLabel += [[]]
                for j in range(17):
                    self.spielerLabel[i] += [Label(master = self.block, text = '', width = weite)]
                    self.spielerLabel[i][j].grid(row=j+1, column=i+1, padx='10', pady='5', sticky='w')

        def checkSpielstand():
            index = self.spieler.index(self.aktuellerSpieler)
            gemacht = self.aktuellerSpieler.getGemacht()
            punktzahlen = self.aktuellerSpieler.getEreignisse()
            for i in range(13):
                if gemacht[i]:
                    self.spielerLabel[index][i].config(text=str(punktzahlen[i]))
                    self.spielerLabel[index][i].config(fg='black')
                    self.kategorien[i]['state'] = DISABLED
                else:
                    self.spielerLabel[index][i].config(text='')
                    self.kategorien[i]['state'] = NORMAL
            checkSumme()

        def neu():
            var1.set(0)
            var2.set(0)
            var3.set(0)
            var4.set(0)
            var5.set(0)

        def checkSumme():
            index = self.spieler.index(self.aktuellerSpieler)
            summen = self.aktuellerSpieler.getSummen()
            for i in range(13, 17):
                self.spielerLabel[index][i].config(text=str(summen[i-13]))

        setSpielerLabel()
        weite = '20'
        self.labelRundeName = Label(master = self.fenster, text = 'Runde', width = weite)
        self.labelRundeName.grid(row=0, column=0, padx='5', pady='5', sticky='w')
        self.labelRunde = Label(master = self.fenster, text = '0', width = weite)
        self.labelRunde.grid(row=0, column=1, padx='5', pady='5', sticky='w')
        self.labelName = Label(master = self.fenster, text = 'Name', width = weite)
        self.labelName.grid(row=0, column=2, padx='5', pady='5', sticky='w')
        self.labelNameSpieler = Label(master = self.fenster, text = self.namen[0], width = weite)
        self.labelNameSpieler.grid(row=0, column=3, padx='5', pady='5', sticky='w')
        self.labelWuerfel1 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel1.grid(row=1, column=0, padx='5', pady='5', sticky='w')
        self.labelWuerfel2 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel2.grid(row=1, column=1, padx='5', pady='5', sticky='w')
        self.labelWuerfel3 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel3.grid(row=1, column=2, padx='5', pady='5', sticky='w')
        self.labelWuerfel4 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel4.grid(row=1, column=3, padx='5', pady='5', sticky='w')
        self.labelWuerfel5 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel5.grid(row=1, column=4, padx='5', pady='5', sticky='w')

        self.labelWuerfel11 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel11.grid(row=1, column=0, padx='5', pady='5', sticky='w')
        self.labelWuerfel21 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel21.grid(row=1, column=1, padx='5', pady='5', sticky='w')
        self.labelWuerfel31 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel31.grid(row=1, column=2, padx='5', pady='5', sticky='w')
        self.labelWuerfel41 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel41.grid(row=1, column=3, padx='5', pady='5', sticky='w')
        self.labelWuerfel51 = Label(master = self.fenster, text = '0', width = weite)
        self.labelWuerfel51.grid(row=1, column=4, padx='5', pady='5', sticky='w')

        def aktualisiereWuerfel():
            wuerfelZeichen = ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"]
            labelWuerfel = [self.labelWuerfel1, self.labelWuerfel2, self.labelWuerfel3, self.labelWuerfel4, self.labelWuerfel5]
            labelWuerfel1 = [self.labelWuerfel11, self.labelWuerfel21, self.labelWuerfel31, self.labelWuerfel41, self.labelWuerfel51]
            for i in range(5):
                zahl = int(labelWuerfel[i].cget('text'))
                labelWuerfel1[i].config(text=wuerfelZeichen[zahl-1])
                labelWuerfel1[i].config(font=('Arial', 15))

        var1 = IntVar()
        Checkbutton(master=self.fenster, text='', variable=var1).grid(row=2, column=0, sticky=S)
        var2 = IntVar()
        Checkbutton(master=self.fenster, text='', variable=var2).grid(row=2, column=1, sticky=S)
        var3 = IntVar()
        Checkbutton(master=self.fenster, text='', variable=var3).grid(row=2, column=2, sticky=S)
        var4 = IntVar()
        Checkbutton(master=self.fenster, text='', variable=var4).grid(row=2, column=3, sticky=S)
        var5 = IntVar()
        Checkbutton(master=self.fenster, text='', variable=var5).grid(row=2, column=4, sticky=S)

        button = Button(master = self.fenster, text='würfeln', command = buttonWuerfelnClick)
        button.grid(row=3, column=2, sticky='s')

        def button1er():
            self.aktuellerSpieler.setEreignis(0)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def button2er():
            self.aktuellerSpieler.setEreignis(1)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def button3er():
            self.aktuellerSpieler.setEreignis(2)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def button4er():
            self.aktuellerSpieler.setEreignis(3)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def button5er():
            self.aktuellerSpieler.setEreignis(4)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def button6er():
            self.aktuellerSpieler.setEreignis(5)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def buttonDreier():
            self.aktuellerSpieler.setEreignis(6)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def buttonVierer():
            self.aktuellerSpieler.setEreignis(7)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def buttonFH():
            self.aktuellerSpieler.setEreignis(8)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def buttonklStr():
            self.aktuellerSpieler.setEreignis(9)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def buttongrStr():
            self.aktuellerSpieler.setEreignis(10)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def buttonKniffel():
            self.aktuellerSpieler.setEreignis(11)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()

        def buttonChance():
            self.aktuellerSpieler.setEreignis(12)
            checkSpielstand()
            self.aktuellerSpieler = self.spieler[(self.spieler.index(self.aktuellerSpieler)+1)%self.anzahlSpieler]
            checkWuerfel([])
            neu()


        button1 = Button(master = self.fenster, text='1er', command = button1er)
        button2 = Button(master = self.fenster, text='2er', command = button2er)
        button3 = Button(master = self.fenster, text='3er', command = button3er)
        button4 = Button(master = self.fenster, text='4er', command = button4er)
        button5 = Button(master = self.fenster, text='5er', command = button5er)
        button6 = Button(master = self.fenster, text='6er', command = button6er)
        buttonDreier = Button(master = self.fenster, text='Dreierpasch', command = buttonDreier)
        buttonVierer = Button(master = self.fenster, text='Viererpasch', command = buttonVierer)
        buttonFH = Button(master = self.fenster, text='Full House', command = buttonFH)
        buttonklStr = Button(master = self.fenster, text='Kleine Straße', command = buttonklStr)
        buttongrStr = Button(master = self.fenster, text='Große Straße', command = buttongrStr)
        buttonKniffel = Button(master = self.fenster, text='Kniffel', command = buttonKniffel)
        buttonChance = Button(master = self.fenster, text='Chance', command = buttonChance)

        self.kategorien = [button1, button2, button3, button4, button5, button6, buttonDreier, buttonVierer, buttonFH, buttonklStr, buttongrStr, buttonKniffel, buttonChance]

        x='10'
        y='10'
        button1.grid(row=4, column=0, padx=x, pady=y,sticky='w')
        button2.grid(row=5, column=0, padx=x, pady=y, sticky='w')
        button3.grid(row=6, column=0, padx=x, pady=y ,sticky='w')
        button4.grid(row=7, column=0, padx=x, pady=y, sticky='w')
        button5.grid(row=8, column=0, padx=x, pady=y, sticky='w')
        button6.grid(row=9, column=0, padx=x, pady=y, sticky='w')
        buttonDreier.grid(row=10, column=0, padx=x, pady=y, sticky='w')
        buttonVierer.grid(row=11, column=0, padx=x, pady=y, sticky='w')
        buttonFH.grid(row=12, column=0, padx=x, pady=y, sticky='w')
        buttonklStr.grid(row=13, column=0, padx=x, pady=y, sticky='w')
        buttongrStr.grid(row=14, column=0, padx=x, pady=y, sticky='w')
        buttonKniffel.grid(row=15, column=0, padx=x, pady=y, sticky='w')
        buttonChance.grid(row=16, column=0, padx=x, pady=y, sticky='w')

        self.fenster.mainloop()
        self.block.mainloop()
        
g1 = guimanager()
