from typing import List
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana

class NotSupervisedModel:
    def __init__(self):
        # Lista local de microrregiões (não supervisionadas)
        self.spfMiCS: List[SPFMiC] = []

    def classify(self, example: Example, K: float, updated: int) -> float:
        tipicidades: List[float] = []
        auxSPFMiCs: List[SPFMiC] = []
        isOutlier = True

        for spf in self.spfMiCS:
            distancia = calculaDistanciaEuclidiana(example.getPonto(), spf.getCentroide())
            if distancia <= spf.getRadiusUnsupervised():
                isOutlier = False
                tipicidades.append(
                    spf.calculaTipicidade(example.getPonto(), spf.getN(), K)
                )
                auxSPFMiCs.append(spf)

        if isOutlier:
            return -1.0  # igual ao Java (double)

        maxVal = max(tipicidades)
        indexMax = tipicidades.index(maxVal)

        spfmic = auxSPFMiCs[indexMax]
        index = self.spfMiCS.index(spfmic)

        self.spfMiCS[index].setUpdated(updated)
        return self.spfMiCS[index].getRotulo()

    def removeOldSPFMiCs(self, ts: int, currentTime: int):
        spfMiCSAux = list(self.spfMiCS)  # cópia
        for spf in list(spfMiCSAux):  # iterar sobre cópia
            if (currentTime - spf.getT() > ts) and (currentTime - spf.getUpdated() > ts):
                spfMiCSAux.remove(spf)
        self.spfMiCS = spfMiCSAux
