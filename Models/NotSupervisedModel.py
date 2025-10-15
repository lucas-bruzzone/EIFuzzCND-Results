from typing import List
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana


class NotSupervisedModel:
    def __init__(self):
        self.spfMiCS: List[SPFMiC] = []

    def classify(self, example: Example, K: float, updated: int) -> float:
        tipicidades: List[float] = []
        auxSPFMiCs: List[SPFMiC] = []
        isOutlier = True

        for spf in self.spfMiCS:
            distancia = calculaDistanciaEuclidiana(example, spf.getCentroide())
            if distancia <= spf.getRadiusUnsupervised():
                isOutlier = False
                tipicidades.append(spf.calculaTipicidade(example.getPonto(), spf.getN(), K))
                auxSPFMiCs.append(spf)

        if isOutlier:
            return -1.0

        maxVal = max(tipicidades)
        indexMax = tipicidades.index(maxVal)

        chosen = auxSPFMiCs[indexMax]
        idx = self.spfMiCS.index(chosen)

        self.spfMiCS[idx].setUpdated(updated)
        return self.spfMiCS[idx].getRotulo()

    def removeOldSPFMiCs(self, ts: int, currentTime: int):
        spfMiCSAux = list(self.spfMiCS)
        for spf in self.spfMiCS:
            if (currentTime - spf.getT() > ts) and (currentTime - spf.getUpdated() > ts):
                if spf in spfMiCSAux:
                    spfMiCSAux.remove(spf)
        self.spfMiCS = spfMiCSAux
