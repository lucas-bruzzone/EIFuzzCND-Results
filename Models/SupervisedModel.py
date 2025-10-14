# SupervisedModel.py
import numpy as np
from typing import List, Dict
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.FuzzyFunctions import FuzzyFunctions
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana

class SupervisedModel:
    def __init__(self, dataset: str, caminho: str, fuzzification: float, alpha: float, theta: float, K: int, minWeight: int):
        self.dataset = dataset
        self.caminho = caminho
        self.fuzzification = fuzzification
        self.alpha = alpha
        self.theta = theta
        self.K = K
        self.minWeight = minWeight
        self.knowLabels: List[float] = []
        self.classifier: Dict[float, List[SPFMiC]] = {}   # igual ao Java, por instância

    # ======== Treinamento inicial ========
    def trainInitialModel(self, trainSet: np.ndarray):
        chunk = [Example(row, True) for row in np.asarray(trainSet)]
        examplesByClass = FuzzyFunctions.separateByClasses(chunk)

        for cls, lst in examplesByClass.items():
            if len(lst) > self.K:
                if cls not in self.knowLabels:
                    self.knowLabels.append(cls)

                cntr, u = FuzzyFunctions.fuzzyCMeans(lst, self.K, self.fuzzification)

                spfmics = FuzzyFunctions.separateExamplesByClusterClassifiedByFuzzyCMeans(
                    lst, cntr, u, cls, self.alpha, self.theta, self.minWeight, 0
                )
                self.classifier[cls] = spfmics

    # ======== Classificação de um novo exemplo ========
    def classifyNew(self, ins: List[float], updateTime: int):
        example = Example(ins, False, updateTime)
        return self.classify(self.getAllSPFMiCs(), example, updateTime)

    def classify(self, spfMiCS: List[SPFMiC], example: Example, updatedTime: int):
        tipicidades: List[float] = []
        pertinencias: List[float] = []
        aux: List[SPFMiC] = []
        isOutlier = True
        px = np.array(example.getPonto(), dtype=float)

        for spf in spfMiCS:
            c = np.array(spf.getCentroide(), dtype=float)
            dist = calculaDistanciaEuclidiana(px, c)

            if dist <= spf.getRadiusWithWeight():
                isOutlier = False
                tipicidades.append(spf.calculaTipicidade(px, spf.getN(), self.K))
                pertinencias.append(self.calculaPertinencia(px, c, self.fuzzification))
                aux.append(spf)

        if isOutlier:
            return -1.0

        idx = int(np.argmax(tipicidades))
        chosen = aux[idx]
        j = spfMiCS.index(chosen)

        spfMiCS[j].setUpdated(updatedTime)
        spfMiCS[j].atribuiExemplo(example, pertinencias[idx], 1.0)
        return spfMiCS[j].getRotulo()

    # ======== Pertinência (fórmula clássica do FCM) ========
    @staticmethod
    def calculaPertinencia(x: np.ndarray, centroide: np.ndarray, m: float) -> float:
        dist = np.linalg.norm(x - centroide)
        if dist == 0:
            return 1.0
        return 1.0 / (1.0 + (dist ** 2) / m)

    # ======== Acessores ========
    def getAllSPFMiCs(self) -> List[SPFMiC]:
        out: List[SPFMiC] = []
        for lst in self.classifier.values():
            out.extend(lst)
        return out

    def removeOldSPFMiCs(self, ts: int, currentTime: int):
        for cls, lst in list(self.classifier.items()):
            self.classifier[cls] = [
                spf for spf in lst
                if not (currentTime - spf.getT() > ts and currentTime - spf.getUpdated() > ts)
            ]
