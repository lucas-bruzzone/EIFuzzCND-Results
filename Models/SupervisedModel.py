import numpy as np
from typing import List, Dict
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.FuzzyFunctions import FuzzyFunctions
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana

class SupervisedModel:
    classifier: Dict[float, List[SPFMiC]] = {}

    def __init__(self, dataset: str, caminho: str, fuzzification: float, alpha: float, theta: float, K: int, minWeight: int):
        self.dataset = dataset
        self.caminho = caminho
        self.fuzzification = fuzzification
        self.alpha = alpha
        self.theta = theta
        self.K = K
        self.minWeight = minWeight
        self.knowLabels: List[float] = []

    def trainInitialModel(self, trainSet) -> None:
        chunk: List[Example] = []
        arr = np.asarray(trainSet)
        for i in range(arr.shape[0]):
            ex = Example(arr[i], True, i)
            chunk.append(ex)

        print("\n=== DEBUG: primeiros Examples ===")
        for i, ex in enumerate(chunk[:5]):
            print(f"Example {i}: ponto={ex.getPonto()}, rotulo={ex.getRotuloVerdadeiro()}, time={ex.getTime()}")

        examplesByClass = FuzzyFunctions.separateByClasses(chunk)
        classes: List[float] = list(examplesByClass.keys())

        print("\n=== DEBUG: classes detectadas ===", classes)

        for j in range(len(examplesByClass)):
            cls = float(classes[j])
            lst = examplesByClass[cls]
            if len(lst) > self.K:
                if cls not in self.knowLabels:
                    self.knowLabels.append(cls)

                clusters = FuzzyFunctions.fuzzyCMeans(lst, self.K, self.fuzzification)

                print(f"\n=== DEBUG: Clustering classe {cls} ===")
                print("Centroides:\n", clusters.centroids)
                print("Pertinência (primeiras 5 colunas):\n", clusters.membership[:, :min(5, clusters.membership.shape[1])])

                spfmics = FuzzyFunctions.separateExamplesByClusterClassifiedByFuzzyCMeans(
                    lst, clusters.centroids, clusters.membership, cls, self.alpha, self.theta, self.minWeight, 0
                )

                print(f"SPFMiCs criados para classe {cls}:")
                for sp in spfmics[:3]:
                    print(sp)

                SupervisedModel.classifier[cls] = spfmics

    def classifyNew(self, ins, updateTime: int):
        allSPFMiCSOfClassifier: List[SPFMiC] = []
        allSPFMiCSOfClassifier.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))

        return self.classify(allSPFMiCSOfClassifier, Example(np.asarray(ins), True), updateTime)

    def getAllSPFMiCsFromClassifier(self, classifier: Dict[float, List[SPFMiC]]) -> List[SPFMiC]:
        spfMiCS: List[SPFMiC] = []
        keys: List[float] = list(classifier.keys())
        for i in range(len(classifier)):
            spfMiCS.extend(classifier[keys[i]])
        return spfMiCS

    def classify(self, spfMiCS: List[SPFMiC], example: Example, updateTime: int) -> float:
        tipicidades: List[float] = []
        pertinencia: List[float] = []
        auxSPFMiCs: List[SPFMiC] = []
        isOutlier = True

        for i in range(len(spfMiCS)):
            distancia = calculaDistanciaEuclidiana(example, spfMiCS[i].getCentroide())
            if distancia <= spfMiCS[i].getRadiusWithWeight():
                isOutlier = False
                tipicidades.append(spfMiCS[i].calculaTipicidade(example.getPonto(), spfMiCS[i].getN(), self.K))
                pertinencia.append(SupervisedModel.calculaPertinencia(example.getPonto(), spfMiCS[i].getCentroide(), self.fuzzification))
                auxSPFMiCs.append(spfMiCS[i])

        if isOutlier:
            return -1.0

        maxValTip = max(tipicidades)
        indexMaxTip = tipicidades.index(maxValTip)

        maxValPer = max(pertinencia)

        spfmic = auxSPFMiCs[indexMaxTip]
        index = spfMiCS.index(spfmic)

        spfMiCS[index].setUpdated(updateTime)
        spfMiCS[index].atribuiExemplo(example, maxValPer, 1.0)

        return spfMiCS[index].getRotulo()

    @staticmethod
    def calculaPertinencia(dataPoints: np.ndarray, clusterCentroids: np.ndarray, m: float) -> float:
        dataPoints = np.asarray(dataPoints, dtype=float)
        clusterCentroids = np.asarray(clusterCentroids, dtype=float)
        distance = float(np.sum((dataPoints - clusterCentroids) ** 2))
        return float(np.exp(-distance / float(m)))

    def getAllSPFMiCs(self) -> List[SPFMiC]:
        spfMiCS: List[SPFMiC] = []
        spfMiCS.extend(self.getAllSPFMiCsFromClassifier(SupervisedModel.classifier))
        return spfMiCS

    def trainNewClassifier(self, chunk: List[Example], t: int) -> List[Example]:
        newChunk: List[Example] = []
        examplesByClass = FuzzyFunctions.separateByClasses(chunk)
        classes: List[float] = list(examplesByClass.keys())
        classifier_local: Dict[float, List[SPFMiC]] = {}

        for j in range(len(examplesByClass)):
            cls = float(classes[j])
            lst = examplesByClass[cls]
            if len(lst) >= self.K * 2:
                if cls not in self.knowLabels:
                    self.knowLabels.append(cls)
                clusters = FuzzyFunctions.fuzzyCMeans(lst, self.K, self.fuzzification)
                spfmics = FuzzyFunctions.separateExamplesByClusterClassifiedByFuzzyCMeans(
                    lst, clusters.centroids, clusters.membership, cls, self.alpha, self.theta, self.minWeight, t
                )
                classifier_local[cls] = spfmics
            else:
                newChunk.extend(lst)

        return newChunk

    def removeOldSPFMiCs(self, ts: int, currentTime: int) -> None:
        for cls, spfMiCSatuais in SupervisedModel.classifier.items():
            spfMiCSAux = list(spfMiCSatuais)
            for spf in spfMiCSatuais:
                if (currentTime - spf.getT() > ts) and (currentTime - spf.getUpdated() > ts):
                    try:
                        spfMiCSAux.remove(spf)
                    except ValueError:
                        pass
            SupervisedModel.classifier[cls] = spfMiCSAux