import numpy as np
import pandas as pd

from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from Structs.Example import Example
from Models.NotSupervisedModel import NotSupervisedModel
from Output.HandlesFiles import HandlesFiles
from ConfusionMatrix.ConfusionMatrix import ConfusionMatrix, Metrics
from FuzzyFunctions.FuzzyFunctions import FuzzyFunctions


class OnlinePhase:
    def __init__(self, caminho, supervisedModel, latencia, tChunk, T, kShort, phi, ts, minWeight, percentLabeled):
        self.caminho = caminho
        self.supervisedModel = supervisedModel
        self.latencia = latencia
        self.tChunk = tChunk
        self.T = T
        self.kShort = kShort
        self.phi = phi
        self.ts = ts
        self.minWeight = minWeight
        self.notSupervisedModel = NotSupervisedModel()
        self.percentLabeled = percentLabeled

        self.results = []
        self.novelties = []
        self.existNovelty = False
        self.nPCount = 100.0
        self.divisor = 1000
        self.tamConfusion = 0

    def initialize(self, dataset: str):
        try:
            # Carrega CSV equivalente ao "-instances.arff" do Java
            df = pd.read_csv(self.caminho + dataset + "-instances.csv")
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values

            # Estruturas auxiliares
            esperandoX = X
            esperandoY = y_true
            nExeTemp = 0
            tempoLatencia = 0

            # Matrizes de confusão
            confusionMatrix = ConfusionMatrix()
            confusionMatrixOriginal = ConfusionMatrix()
            metrics_list = []
            append = False

            # Memórias
            unkMem = []
            labeledMem = []
            trueLabels = set()

            # Loop principal
            for tempo in range(len(X)):
                tempoLatencia += 1

                # Constrói Example do item corrente
                row = X[tempo]
                exemplo = Example(np.concatenate([row, [y_true[tempo]]]), True, tempo)

                # Classificação supervisionada
                rotulo = self.supervisedModel.classifyNew(row, tempo)
                exemplo.setRotuloClassificado(rotulo)

                # Gestão do conjunto de classes verdadeiras
                if (exemplo.getRotuloVerdadeiro() not in trueLabels) or (confusionMatrixOriginal.getNumberOfClasses() != self.tamConfusion):
                    trueLabels.add(exemplo.getRotuloVerdadeiro())
                    self.tamConfusion = confusionMatrixOriginal.getNumberOfClasses()
                    # (no Java poderia salvar/atualizar matriz original aqui)

                # Se outlier (-1), tenta classificar com não supervisionado
                if rotulo == -1 or rotulo == -1.0:
                    rotulo_ns = self.notSupervisedModel.classify(exemplo, float(self.supervisedModel.K), tempo)
                    exemplo.setRotuloClassificado(rotulo_ns)

                    # Continua desconhecido → vai para unkMem e tenta ND
                    if rotulo_ns == -1 or rotulo_ns == -1.0:
                        unkMem.append(exemplo)
                        if len(unkMem) >= self.T:
                            unkMem = self.multiClassNoveltyDetection(
                                unkMem, tempo, confusionMatrix, confusionMatrixOriginal
                            )

                # Atualiza resultados e matriz de confusão
                self.results.append(exemplo)
                confusionMatrix.addInstance(exemplo.getRotuloVerdadeiro(), exemplo.getRotuloClassificado())
                # confusionMatrixOriginal.addInstance(... ) // igual ao Java, está comentado

                # Lógica de latência (espelhando Java: usa acumulador tempoLatencia e nExeTemp)
                if tempoLatencia >= self.latencia:
                    # Com probabilidade ou se labeledMem está vazia, push um rotulado da fila "esperando"
                    if (np.random.rand() < self.percentLabeled) or (len(labeledMem) == 0):
                        # Em Java: Example(esperandoTempo.get(nExeTemp).toDoubleArray(), true, tempo)
                        if nExeTemp < len(esperandoX):
                            arr = np.concatenate([esperandoX[nExeTemp], [esperandoY[nExeTemp]]])
                            labeledExample = Example(arr, True, tempo)
                            labeledMem.append(labeledExample)

                    # Treina incrementalmente quando acumula tChunk
                    if len(labeledMem) >= self.tChunk:
                        labeledMem = self.supervisedModel.trainNewClassifier(labeledMem, tempo)
                        labeledMem.clear()

                    nExeTemp += 1
                    # OBS: Java NÃO zera tempoLatencia (após alcançar latência roda sempre esse bloco).
                    # Para espelhar exatamente, NÃO resetamos.

                # Remoção de SPFMiCs antigos
                self.supervisedModel.removeOldSPFMiCs(self.latencia + self.ts, tempo)
                # notSupervisedModel.removeOldSPFMiCs(...) está comentado no Java

                # Remove/filtra unknowns conforme regra Java (mantém ct - time >= ts)
                unkMem = self.removeOldUnknown(unkMem, self.ts, tempo)

                # Métricas periódicas
                if (tempo > 0) and (tempo % self.divisor == 0):
                    # No Java: confusionMatrix.mergeClasses(confusionMatrix.getClassesWithNonZeroCount());
                    confusionMatrix.mergeClasses(confusionMatrix.getClassesWithNonZeroCount())
                    metrics = confusionMatrix.calculateMetrics(tempo, confusionMatrix.countUnknow(), self.divisor)
                    print(f"Tempo:{tempo} Acurácia: {metrics.getAccuracy()} Precision: {metrics.getPrecision()}")
                    metrics_list.append(metrics)
                    self.novelties.append(1.0 if self.existNovelty else 0.0)
                    self.existNovelty = False

            # Salva métricas/resultados (como no Java)
            for metrica in metrics_list:
                HandlesFiles.salvaMetrics(
                    int(metrica.getTempo() / self.divisor),
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

        except Exception as e:
            raise RuntimeError(e)

    def multiClassNoveltyDetection(self, listaDesconhecidos, tempo, confusionMatrix, confusionMatrixOriginal):
        # Espelha o Java: roda apenas se #desconhecidos > kShort
        if len(listaDesconhecidos) > self.kShort:

            # FCM nos desconhecidos
            cntr, u = FuzzyFunctions.fuzzyCMeans(listaDesconhecidos, self.kShort, self.supervisedModel.fuzzification)

            # Silhuetas (em Python usamos a matriz de pertinência)
            silhuetas = FuzzyFunctions.fuzzySilhouette(u, listaDesconhecidos, self.supervisedModel.alpha)

            # Tamanho de cada cluster (equivalente a centroid.getPoints().size())
            winners = np.argmax(u, axis=0)                 # vencedor por exemplo
            counts_per_cluster = np.bincount(winners, minlength=self.kShort)

            # Índices de clusters válidos (silhueta > 0 e tamanho >= minWeight)
            silhuetasValidas = [i for i, s in enumerate(silhuetas) if (s > 0) and (counts_per_cluster[i] >= self.minWeight)]

            # SPFMiCs para os desconhecidos (igual ao Java: versão "newSeparate...")
            sfMiCS = FuzzyFunctions.newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
                listaDesconhecidos, cntr, u, -1.0, self.supervisedModel.alpha,
                self.supervisedModel.theta, self.minWeight, tempo
            )

            # Todos os microrregiões conhecidos
            sfmicsConhecidos = self.supervisedModel.getAllSPFMiCs()

            # Para cada cluster válido, decidir se é conhecido (FR <= phi) ou novidade
            for i in range(len(cntr)):

                # Em Java: if (silhuetasValidas.contains(i) && !sfMiCS.get(i).isNull())
                if (i in silhuetasValidas) and (i < len(sfMiCS)) and (sfMiCS[i] is not None):
                    # Calcula FRs (Java faz (di+dj)/dist e depois adiciona (di+dj)/dist → acaba sendo a distância)
                    frs = []
                    for known in sfmicsConhecidos:
                        di = known.getRadiusND()
                        dj = sfMiCS[i].getRadiusND()
                        distancia = calculaDistanciaEuclidiana(known.getCentroide(), sfMiCS[i].getCentroide())
                        # replicando exatamente a álgebra do Java:
                        denom = (di + dj) / (distancia if distancia != 0 else 1e-12)
                        frs.append((di + dj) / denom)  # isso resulta em "distancia"

                    if frs:
                        minFr = min(frs)
                        indexMinFr = frs.index(minFr)

                        # Exemplos pertencentes ao cluster i (equivalente a centroid.get(i).getPoints())
                        exemplos_cluster_i = [ex for idx, ex in enumerate(listaDesconhecidos) if winners[idx] == i]

                        if minFr <= self.phi:
                            # Atribui rótulo do conhecido mais próximo
                            sfMiCS[i].setRotulo(sfmicsConhecidos[indexMinFr].getRotulo())

                            # Atualiza ConfusionMatrix e remove do buffer de desconhecidos
                            rotulos = {}
                            for ex in exemplos_cluster_i:
                                # remove (como no Java)
                                if ex in listaDesconhecidos:
                                    listaDesconhecidos.remove(ex)

                                trueLabel = ex.getRotuloVerdadeiro()
                                predictedLabel = sfMiCS[i].getRotulo()
                                OnlinePhase.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                # OnlinePhase.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrixOriginal)

                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            # Maioria dos rótulos verdadeiros
                            if rotulos:
                                keys = list(rotulos.keys())
                                maiorRotulo = max(keys, key=lambda k: rotulos[k])
                            else:
                                maiorRotulo = -1.0

                            # Só promove a conhecido se maioria bate com rótulo atribuído
                            if maiorRotulo == sfMiCS[i].getRotulo():
                                sfMiCS[i].setRotuloReal(maiorRotulo)
                                self.notSupervisedModel.spfMiCS.append(sfMiCS[i])

                        else:
                            # NOVELTY
                            self.existNovelty = True
                            sfMiCS[i].setRotulo(self.generateNPLabel())

                            rotulos = {}
                            for ex in exemplos_cluster_i:
                                if ex in listaDesconhecidos:
                                    listaDesconhecidos.remove(ex)

                                trueLabel = ex.getRotuloVerdadeiro()
                                predictedLabel = sfMiCS[i].getRotulo()
                                OnlinePhase.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix)
                                # OnlinePhase.updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrixOriginal)

                                rotulos[trueLabel] = rotulos.get(trueLabel, 0) + 1

                            if rotulos:
                                keys = list(rotulos.keys())
                                maiorRotulo = max(keys, key=lambda k: rotulos[k])
                            else:
                                maiorRotulo = -1.0

                            sfMiCS[i].setRotuloReal(maiorRotulo)
                            self.notSupervisedModel.spfMiCS.append(sfMiCS[i])

        return listaDesconhecidos

    # ATENÇÃO: esta função espelha literalmente a implementação Java
    def removeOldUnknown(self, unkMem, ts, ct):
        newUnkMem = []
        for ex in unkMem:
            # Java mantém exemplos cujo (ct - time) >= ts
            if (ct - ex.getTime()) >= ts:
                newUnkMem.append(ex)
        return newUnkMem

    def generateNPLabel(self):
        self.nPCount += 1.0
        return self.nPCount

    @staticmethod
    def updateConfusionMatrix(trueLabel, predictedLabel, confusionMatrix: ConfusionMatrix):
        confusionMatrix.addInstance(trueLabel, predictedLabel)
        confusionMatrix.updateConfusionMatrix(trueLabel)

    def getTamConfusion(self):
        return self.tamConfusion

    def setTamConfusion(self, tam):
        self.tamConfusion = tam
