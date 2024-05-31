import math

import numpy as np


def anzahlAugenzahlen(wuerfel):
    augenzahlen = []
    for i in range(1,7):
        augenzahlen += [wuerfel.count(i)]
    return augenzahlen

def berechneWuerfelBehalten(wuerfel, wuerfeUebrig, kategorien):
    erwartungswerte = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    behalten = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    oben = kategorien[:6]
    unten = kategorien[6:]
    for i in range(6):
        if kategorien[i] == 0:
            behalten[i] = wuerfel.count(i+1)*[i+1]
            summe =wuerfel.count(i+1)*(i+1)
            erwartungswerte[i] = wuerfeUebrig*(5-wuerfel.count(i+1))*1/6*(i+1)+summe
            #print(erwartungswerte[i], behalten[i])
        else:
            behalten[i] = []

    if kategorien[6] == 0:
        maxErwartungswert = 0
        maxBehalten = []
        for i in range(6):
            augenzahlen = anzahlAugenzahlen(wuerfel)
            anzahl = augenzahlen[i]
            fehlen = 3 - anzahl
            anzahlWuerfel = 5 - anzahl
            augenzahl = i+1
            if fehlen <= 0:
                behaltenW = [augenzahl]*3
                erwartungswert = 3*augenzahl
                augenzahlen[i] -= 3
                if augenzahlen[4] != 0:
                    behaltenW += [5]*augenzahlen[4]
                    erwartungswert += augenzahlen[4]*5
                if augenzahlen[5] != 6:
                    behaltenW += [6]*augenzahlen[5]
                    erwartungswert += augenzahlen[5]*6
                erwartungswert += 3.5*(5-len(behaltenW))
                if erwartungswert > maxErwartungswert:
                    maxErwartungswert = erwartungswert
                    maxBehalten = behaltenW
            elif fehlen == 1:
                behaltenW = [augenzahl]*2
                augenzahlen[i] -= 2
                erwartungswert = (1-((5/6)**(anzahlWuerfel*wuerfeUebrig)))*(3*augenzahl+2*3.5)
                if erwartungswert > maxErwartungswert:
                    maxErwartungswert = erwartungswert
                    maxBehalten = behaltenW
            elif fehlen == 2:
                behaltenW = [augenzahl]
                augenzahlen[i] -= 1
                erwartungswert = (1-(((5/6)**(anzahlWuerfel*wuerfeUebrig))+anzahlWuerfel*wuerfeUebrig*1/6*(5/6)**(anzahlWuerfel*wuerfeUebrig-1)))*(3*augenzahl+2*3.5)
                if erwartungswert > maxErwartungswert:
                    maxErwartungswert = erwartungswert
                    maxBehalten = behaltenW
        erwartungswerte[6] = maxErwartungswert
        behalten[6] = maxBehalten
        #print('3er Pasch', erwartungswerte[6], behalten[6])

    if kategorien[7] == 0:
        maxErwartungswert = 0
        maxBehalten = []
        for i in range(6):
            augenzahlen = anzahlAugenzahlen(wuerfel)
            anzahl = augenzahlen[i]
            fehlen = 4 - anzahl
            anzahlWuerfel = 5 - anzahl
            augenzahl = i+1
            if fehlen <= 0:
                behaltenW = [augenzahl]*4
                erwartungswert = 4*augenzahl
                augenzahlen[i] -= 4
                if augenzahlen[4] != 0:
                    behaltenW += [5]*augenzahlen[4]
                    erwartungswert += augenzahlen[4]*5
                if augenzahlen[5] != 6:
                    behaltenW += [6]*augenzahlen[5]
                    erwartungswert += augenzahlen[5]*6
                erwartungswert += 3.5*(5-len(behaltenW))
                if erwartungswert > maxErwartungswert:
                    maxErwartungswert = erwartungswert
                    maxBehalten = behaltenW
            elif fehlen == 1:
                behaltenW = [augenzahl]*3
                augenzahlen[i] -= 3
                p1 = (5/6)**(anzahlWuerfel*wuerfeUebrig)
                erwartungswert = (1-p1)*(4*augenzahl+3.5)
                if erwartungswert > maxErwartungswert:
                    maxErwartungswert = erwartungswert
                    maxBehalten = behaltenW
            elif fehlen == 2:
                behaltenW = [augenzahl]*2
                augenzahlen[i] -= 2
                p1 = (5/6)**(anzahlWuerfel*wuerfeUebrig)
                p2 = anzahlWuerfel*wuerfeUebrig*1/6*((5/6)**(anzahlWuerfel*wuerfeUebrig-1))
                erwartungswert = (1-(p1+p2))*(4*augenzahl+3.5)
                if erwartungswert > maxErwartungswert:
                    maxErwartungswert = erwartungswert
                    maxBehalten = behaltenW
            elif fehlen == 3:
                behaltenW = [augenzahl]
                augenzahlen[i] -= 1
                p1 = (5/6)**(anzahlWuerfel*wuerfeUebrig)
                p2 = anzahlWuerfel*wuerfeUebrig*1/6*((5/6)**(anzahlWuerfel*wuerfeUebrig-1))
                p3 = anzahlWuerfel*wuerfeUebrig*1/6*1/6*((5/6)**(anzahlWuerfel*wuerfeUebrig-2))
                erwartungswert = (1-(p1+p2+p3))*(4*augenzahl+3.5)
                if erwartungswert > maxErwartungswert:
                    maxErwartungswert = erwartungswert
                    maxBehalten = behaltenW
        erwartungswerte[7] = maxErwartungswert
        behalten[7] = maxBehalten
        #print('4er Pasch', erwartungswerte[7], behalten[7])

    if kategorien[8] == 0:
        augenzahlen = anzahlAugenzahlen(wuerfel)
        if 5 in augenzahlen:
            behalten[8] = 3*[(augenzahlen.index(5)+1)]
            p1 = 5/36
            p2 = 0
            p3 = 0
            if wuerfeUebrig == 2:
                p2 = (1-5/36)*5/36
                p3 = (1-5/36)*1/6
            erwartungswerte[8] = (p1+p2+p3)*25
        elif 4 in augenzahlen:
            behalten[8] = 3*[(augenzahlen.index(4)+1)]+ [(augenzahlen.index(1)+1)]
            p1 = 1/6*wuerfeUebrig
            erwartungswerte[8] = p1*25
        elif 3 in augenzahlen and 2 in augenzahlen:
            behalten[8] = 3*[(augenzahlen.index(3)+1)]+ 2*[(augenzahlen.index(2)+1)]
            erwartungswerte[8] = 25
        elif 3 in augenzahlen and 1 in augenzahlen:
            behalten[8] = 3*[(augenzahlen.index(3)+1)]+ [(augenzahlen.index(1)+1)]
            p1 = 1/6*wuerfeUebrig
            erwartungswerte[8] = p1*25
        elif augenzahlen.count(2) == 2:
            zahlen = []
            for i in range(6):
                if augenzahlen[i] == 2:
                    zahlen += [i+1]
            behalten[8] = 2*[zahlen[0]]+2*[zahlen[1]]
            p1 = 2/6*wuerfeUebrig
            erwartungswerte[8] = p1*25
        elif augenzahlen.count(2) == 1:
            behalten[8] = 2*[augenzahlen.index(2)]
            p1 = 1/6+5/36
            p2 = (1-1/6)*5/216
            erwartungswerte[8] = (p1+p2)*25
        else:
            behalten[8] = [wuerfel[0]]
        #print('Full House', erwartungswerte[8], behalten[8])

    if kategorien[9] == 0:
        if 1 in wuerfel and 2 in wuerfel and 3 in wuerfel and 4 in wuerfel:
            behalten[9] += [1, 2, 3, 4]
            erwartungswerte[9] = 30
        elif 2 in wuerfel and 3 in wuerfel and 4 in wuerfel and 5 in wuerfel:
            behalten[9] = [2, 3, 4, 5]
            erwartungswerte[9] = 30
        elif 3 in wuerfel and 4 in wuerfel and 5 in wuerfel and 6 in wuerfel:
            behalten[9] = [3, 4, 5, 6]
            erwartungswerte[9] = 30
        else:
            einlinge = []
            for i in range(1, 7):
                if i in wuerfel:
                    einlinge += [1]
                else:
                    einlinge += [0]
            if einlinge[3:] == [1, 1, 1]:
                behalten[9] = [4, 5, 6]
                erwartungswerte[9] = 1/6*30*wuerfeUebrig
            for i in range(3):
                if einlinge[i:i+3] == [1, 1, 1]:
                    behalten[9] = [i+1, i+2, i+3]
                    erwartungswerte[9] = 1/6*30*wuerfeUebrig
                    break
                elif einlinge[i:i+4] == [1, 1, 0, 1]:
                    behalten[9] = [i+1, i+2, i+4]
                    erwartungswerte[9] = 1/6*30*wuerfeUebrig
                    break
                elif einlinge[i:i+4] == [1, 0, 1, 1]:
                    behalten[9] = [i+1, i+3, i+4]
                    erwartungswerte[9] = 1/6*30*wuerfeUebrig
                    break
            if erwartungswerte[9] == 0:
                for i in range(4):
                    if einlinge[i:i+3] == [1, 1, 1]:
                        behaltenW = [i+1, i+2, i+3]
                        if 1 in behaltenW:
                            behaltenW.remove(1)
                        if 6 in behaltenW:
                            behaltenW.remove(6)
                        behalten += [behaltenW]
                        if len(behaltenW) == 2:
                            erwartungswerte[9] = 1/6*1/3*30*wuerfeUebrig
                        elif len(behaltenW) == 3:
                            erwartungswerte[9] = 1/6*30*wuerfeUebrig
                        break
                for i in range(5):
                    if einlinge[i:i+2] == [1, 1]:
                        behaltenW = [i+1, i+2]
                        if 1 in behaltenW:
                            behaltenW.remove(1)
                        if 6 in behaltenW:
                            behaltenW.remove(6)
                        behalten[9] = behaltenW
                        if len(behaltenW) == 2:
                            erwartungswerte[9] = 1/6*1/3*30*wuerfeUebrig
                        else:
                            erwartungswerte[9] = 1/6*1/6*1/3*30*wuerfeUebrig
                        break
                    if i != 4:
                        if einlinge[i:i+3] == [1, 0, 1]:
                            behaltenW = [i+1, i+2]
                            if 1 in behaltenW:
                                behaltenW.remove(1)
                            if 6 in behaltenW:
                                behaltenW.remove(6)
                            behalten[9] = behaltenW
                            if len(behaltenW) == 2:
                                erwartungswerte[9] = 1/6*1/3*30*wuerfeUebrig
                            else:
                                erwartungswerte[9] = 1/6*1/6*1/3*30*wuerfeUebrig
                            break
            if erwartungswerte[9] == 0:
                if 3 in wuerfel:
                    behalten[9] = [3]
                    erwartungswerte[9] = 1/2*1/3+1/6*30*wuerfeUebrig
                if 4 in wuerfel:
                    behalten[9] = [4]
                    erwartungswerte[9] = 1/2*1/3+1/6*30*wuerfeUebrig
    #print('kleine Straße', erwartungswerte[9], behalten[9])

    if kategorien[10] == 0:
        einlinge = []
        for i in range(1, 7):
            if i in wuerfel:
                einlinge += [1]
            else:
                einlinge += [0]
        if einlinge[:5] == [1, 1, 1, 1, 1]:
            behalten[10] = [1, 2, 3, 4, 5]
            erwartungswerte[10] = 40
        elif einlinge[1:] == [1, 1, 1, 1, 1]:
            behalten[10] = [2, 3, 4, 5, 6]
            erwartungswerte[10] = 40
        else:
            for i in range(3):
                if einlinge[i:i+4] == [1, 1, 1, 1]:
                    behalten[10] = [i+1, i+2, i+3, i+4]
                    x = 1
                    if i == 1:
                        x = 2
                    erwartungswerte[10] = 40*1/6*x*wuerfeUebrig
            if erwartungswerte[10] == 0:
                behaltenW = []
                for i in range(1, 6):
                    if einlinge[i] == 1:
                        behaltenW += [i+1]
                    if len(behaltenW) == 3:
                        erwartungswerte[10] = 1/3*1/6*wuerfeUebrig*40
                    if len(behaltenW) == 2:
                        erwartungswerte[10] = 2/3*1/3*1/6*wuerfeUebrig*40
                    if len(behaltenW) == 1:
                        erwartungswerte[10] = 5/6*1/2*1/3*1/6*wuerfeUebrig*40
                behalten[10] = behaltenW
        #print('große Straße', erwartungswerte[10], behalten[10])

    if kategorien[11] == 0:
        augenzahlen = anzahlAugenzahlen(wuerfel)
        index = augenzahlen.index(max(augenzahlen))
        anzahl = augenzahlen[index]
        behalten[11] = anzahl*[index+1]
        erwartungswerte[11] = ((1/6)**(5-anzahl))*50
    #print('Kniffel', erwartungswerte[11], behalten[11])

    if kategorien.count(1) == 12:
        if kategorien[12] == 0:
            if wuerfeUebrig == 2:
                behalten[12] = wuerfel.count(5)*[5]+wuerfel.count(6)*[6]
                laenge = wuerfel.count(5) + wuerfel.count(6)
                erwartungswerte[12] = wuerfel.count(5)*5 + wuerfel.count(6)*6 + (5-laenge)*3.5
            elif wuerfeUebrig == 1:
                behalten[12] = wuerfel.count(4)*[4]+wuerfel.count(5)*[5]+wuerfel.count(6)*[6]
                laenge = wuerfel.count(4) + wuerfel.count(5) + wuerfel.count(6)
                erwartungswerte[12] = wuerfel.count(4)*4 + wuerfel.count(5)*5 + wuerfel.count(6)*6 + (5-laenge)*3.5
            #print('Chance', erwartungswerte[12], behalten[12])
            #print(erwartungswerte, behalten)

    #print(erwartungswerte, behalten)
    index = erwartungswerte.index(max(erwartungswerte))
    #print(erwartungswerte[index], behalten[index])
    return erwartungswerte[index], behalten[index]

def normalisiereWuerfel(wuerfel):
    wuerfelN = np.array([0,0,0,0,0], dtype='float')
    for i in range(5):
        wuerfelN[i] = wuerfel[i]/10
    #print(wuerfelN)
    return wuerfelN

def wuerfelWiederherstellen(wuerfel):
    wuerfelN = np.array([0,0,0,0,0], dtype='int')
    for i in range(5):
        wuerfelN[i] = wuerfel[i]*10
    #print(wuerfelN)
    return wuerfelN
