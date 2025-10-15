import random
from typing import List, Set
import numpy as np

from Models.SupervisedModel import SupervisedModel
from Models.NotSupervisedModel import NotSupervisedModel
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC
from FuzzyFunctions.FuzzyFunctions import FuzzyFunctions
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from ConfusionMatrix.ConfusionMatrix import ConfusionMatrix
from ConfusionMatrix.Metrics import Metrics
from Output.HandlesFiles import HandlesFiles

try:
    from scipy.io import arff
    import pandas as pd
except ImportError:
    arff = None
    pd = None


class OnlinePhase:
    def __init__(self, caminho: str, supervisedModel: SupervisedModel, latencia: int, tChunk: int, T: int,
                 kShort: int, phi: float, ts: int, minWeight: int, percentLabeled: float):
        self.kShort = kShort
        self.ts = ts
        self.minWeight = minWeight
        self.T = T
        self.caminho = caminho
        self.latencia = latencia
        self.tChunk = tChunk
        self.supervisedModel = supervisedModel
        self.notSupervisedModel = NotSupervisedModel()
        self.phi = phi
        self.existNovelty = False
        self.nPCount = 100.0
        self.novelties: List[float] = []
        self.percentLabeled = percentLabeled
        self.results: List[Example] = []
        self.divisor = 1000
        self.tamConfusion = 0

    def initialize(self, dataset: str):
        # Intermediária
        esperandoTempo = None
        nExeTemp = 0

        # ConfusionMatrix
        confusionMatrix = ConfusionMatrix()
        confusionMatrixOriginal = ConfusionMatrix()
        append = False
        listaMetricas: List[Metrics] = []

        # Carrega .arff de forma equivalente ao DataSource do Weka
        # Java: caminho + dataset + "-instances.arff"
        path = f"{self.caminho}{dataset}-instances.arff"
        if arff is None or pd is None:
            raise RuntimeError("scipy.io.arff e pandas são necessários para ler ARFF.")

        data_np, meta = arff.loadarff(path)
        df = pd.DataFrame(data_np)
        # Assume que a última coluna é a classe (como data.setClassIndex(numAttributes()-1))
        values = df.values  # shape: (n, d)
        data = values  # usando array numpy como 'Instances'

        # Intermediária
        esperandoTempo = data
        labeledMem: List[Example] = []
        trueLabels: Set[float] = set()
        unkMem: List[Example] = []

        tempoLatencia = 0
        for tempo in range(data.shape[0]):
            ins_array = np.asarray(data[tempo], dtype=float)

            # Cria Example com rótulo verdadeiro (última coluna)
            exemplo = Example(ins_array, True, tempo)

            # classifica com modelo supervisionado (Java passa Instance; aqui passamos o vetor)
            rotulo = self.supervisedModel.classifyNew(ins_array, tempo)
            exemplo.setRotuloClassificado(rotulo)

            if (exemplo.getRotuloVerdadeiro() not in trueLabels) or \
               (confusionMatrixOriginal.getNumberOfClasses() != self.tamConfusion):
                trueLabels.add(exemplo.getRotuloVerdadeiro())
                self.tamConfusion = confusionMatrixOriginal.getNumberOfClasses()

            if rotulo == -1 or rotulo == -1.0:
                rotulo = self.notSupervisedModel.classify(exemplo, self.supervisedModel.K, tempo)
                exemplo.setRotuloClassificado(rotulo)
                if rotulo == -1 or rotulo == -1.0:
                    unkMem.append(exemplo)
                    if len(unkMem) >= self.T:
                        unkMem = self.multiClassNoveltyDetection(unkMem, tempo, confusionMatrix, confusionMatrixOriginal)

            self.results.append(exemplo)
            confusionMatrix.addInstance(exemplo.getRotuloVerdadeiro(), exemplo.getRotuloClassificado())
            # confusionMatrixOriginal.addInstance(...)

            tempoLatencia += 1
            if tempoLatencia >= self.latencia:
                if (random.random() < self.percentLabeled) or (len(labeledMem) == 0):
                    labeledExample = Example(esperandoTempo[nExeTemp], True, tempo)
                    labeledMem.append(labeledExample)

                if len(labeledMem) >= self.tChunk:
                    labeledMem = self.supervisedModel.trainNewClassifier(labeledMem, tempo)
                    labeledMem.clear()

                nExeTemp += 1

            self.supervisedModel.removeOldSPFMiCs(self.latencia + self.ts, tempo)
            # self.notSupervisedModel.removeOldSPFMiCs(self.latencia + self.ts, tempo)
            self.removeOldUnknown(unkMem, self.ts, tempo)  # manter o “bug” de ignorar retorno, como no Java

            # Métricas a cada 'divisor'
            if (tempo > 0) and (tempo % int(self.divisor) == 0):
                confusionMatrix.mergeClasses(confusionMatrix.getClassesWithNonZeroCount())
                metrics: Metrics = confusionMatrix.calculateMetrics(tempo, confusionMatrix.countUnknow(), self.divisor)
                print(f"Tempo:{tempo} Acurácia: {metrics.getAccuracy()} Precision: {metrics.getPrecision()}")
                listaMetricas.append(metrics)

                if self.existNovelty:
                    self.novelties.append(1.0)
                    self.existNovelty = False
                else:
                    self.novelties.append(0.0)

        for metrica in listaMetricas:
            # no Java: metrica.getTempo()/divisor depois cast para int
            tempo_idx = int(metrica.getTempo() / self.divisor)
            HandlesFiles.salvaMetrics(
                tempo_idx,
                metrica.getAccuracy(),
                metrica.getPrecision(),
                metrica.getRecall(),
                metrica.getF1Score(),
                dataset,
                self.latencia,
                self.percentLabeled,
                metrica.getUnkMem(),
                metrica.getUnknownRate(),
                append
            )
            append = True

        HandlesFiles.salvaNovidades(self.novelties, dataset, self.latencia, self.percentLabeled)
        HandlesFiles.salvaResultados(self.results, dataset, self.latencia, self.percentLabeled)

    def multiClassNoveltyDetection(self, listaDesconhecidos: List[Example], tempo: int,
                                   confusionMatrix: ConfusionMatrix,
                                   confusionMatrixOriginal: ConfusionMatrix) -> List[Example]:
        if len(listaDesconhecidos) > self.kShort:
            clusters = FuzzyFunctions.fuzzyCMeans(listaDesconhecidos, self.kShort, self.supervisedModel.fuzzification)
            centroides = clusters.getClusters()
            silhuetas = FuzzyFunctions.fuzzySilhouette(clusters, listaDesconhecidos, self.supervisedModel.alpha)
            silhuetasValidas: List[int] = []

            for i in range(len(silhuetas)):
                if (silhuetas[i] > 0) and (len(centroides[i].getPoints()) >= self.minWeight):
                    silhuetasValidas.append(i)

            sfMiCS: List[SPFMiC] = FuzzyFunctions.newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
                listaDesconhecidos, clusters, -1, self.supervisedModel.alpha, self.supervisedModel.theta,
                self.minWeight, tempo
            )
            sfmicsConhecidos: List[SPFMiC] = self.supervisedModel.getAllSPFMiCs()
            frs: List[float] = []

            for i in range(len(centroides)):
                if (i in silhuetasValidas) and (not sfMiCS[i].isNullFunc()):
                    frs.clear()
                    for j in range(len(sfmicsConhecidos)):
                        di = sfmicsConhecidos[j].getRadiusND()
                        dj = sfMiCS[i].getRadiusND()
                        # Atenção: aqui o Java calcula dist = (di + dj) / distEuclid ; depois usa frs.add((di+dj)/dist)
                        dist_euclid = calculaDistanciaEuclidiana(sfmicsConhecidos[j].getCentroide(), sfMiCS[i].getCentroide())
                        dist = (di + dj) / dist_euclid if dist_euclid != 0 else float('inf')
                        frs.append((di + dj) / dist if dist != 0 else float('inf'))

                    if len(frs) > 0:
                        minFr = min(frs)
                        indexMinFr = frs.index(minFr)

                        if minFr <= self.phi:
                            # Herdar rótulo conhecido
                            sfMiCS[i].setRotulo(sfmicsConhecidos[indexMinFr].getRotulo())
                            examples: List[Example] = centroides[i].getPoints()
                            rotulos = {}  # HashMap<Double, Integer> no Java
                            for j in range(len(examples)):
                                try:
                                    listaDesconhecidos.remove(examples[j])
                                except ValueError:
                                    pass

                                trueLabel = examples[j].getRotuloVerdadeiro()
                                predictedLabel = sfMiCS[i].getRotulo()
                                self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                # self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrixOriginal)

                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            # Maioria dos rótulos verdadeiros
                            maiorValor = -float('inf')
                            maiorRotulo = -1.0
                            for key, val in rotulos.items():
                                if maiorValor < val:
                                    maiorValor = val
                                    maiorRotulo = key

                            if maiorRotulo == sfMiCS[i].getRotulo():
                                sfMiCS[i].setRotuloReal(maiorRotulo)
                                self.notSupervisedModel.spfMiCS.append(sfMiCS[i])
                        else:
                            # Novidade
                            self.existNovelty = True
                            sfMiCS[i].setRotulo(self.generateNPLabel())
                            examples: List[Example] = centroides[i].getPoints()
                            rotulos = {}
                            for j in range(len(examples)):
                                try:
                                    listaDesconhecidos.remove(examples[j])
                                except ValueError:
                                    pass

                                trueLabel = examples[j].getRotuloVerdadeiro()
                                predictedLabel = sfMiCS[i].getRotulo()
                                self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                # self.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrixOriginal)

                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            maiorValor = -float('inf')
                            maiorRotulo = -1.0
                            for key, val in rotulos.items():
                                if maiorValor < val:
                                    maiorValor = val
                                    maiorRotulo = key

                            sfMiCS[i].setRotuloReal(maiorRotulo)
                            self.notSupervisedModel.spfMiCS.append(sfMiCS[i])

        return listaDesconhecidos

    def generateNPLabel(self) -> float:
        self.nPCount += 1
        return self.nPCount

    def removeOldUnknown(self, unkMem: List[Example], ts: int, ct: int) -> List[Example]:
        newUnkMem: List[Example] = []
        for i in range(len(unkMem)):
            if ct - unkMem[i].getTime() >= ts:
                newUnkMem.append(unkMem[i])
        return newUnkMem  # No Java, retorno é ignorado no caller

    @staticmethod
    def updateConfusionMatrix(trueLabel: float, predictedLabel: float, confusionMatrix: ConfusionMatrix):
        confusionMatrix.addInstance(trueLabel, predictedLabel)
        confusionMatrix.updateConfusionMatrix(trueLabel)

    def getTamConfusion(self) -> int:
        return self.tamConfusion

    def setTamConfusion(self, tamConfusion: int):
        self.tamConfusion = tamConfusion
