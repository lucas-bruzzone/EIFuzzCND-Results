import numpy as np
import skfuzzy as fuzz
from typing import List, Dict
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC

class FuzzyFunctions:

    @staticmethod
    def fuzzyCMeans(examples: List[Example], K: int, fuzzification: float):
        """Equivalente ao fuzzyCMeans do Java, retorna centroides e matriz de pertinência"""
        data = np.array([ex.getPonto() for ex in examples]).T  # shape: features x samples
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c=K, m=fuzzification, error=1e-6, maxiter=1000, init=None
        )
        return cntr, u  # centroides, pertinência

    @staticmethod
    def fuzzySilhouette(membership_matrix: np.ndarray, examples: List[Example], alpha: float):
        nExemplos = len(examples)
        silhuetas = []
        for i in range(membership_matrix.shape[0]):  # clusters
            numerador = denominador = 0
            for j in range(nExemplos):
                indexClasse = np.argmax(membership_matrix[:, j])
                if indexClasse == i:
                    apj = 0
                    dqj = []
                    for k in range(nExemplos):
                        if np.argmax(membership_matrix[:, k]) == indexClasse:
                            apj += calculaDistanciaEuclidiana(examples[j].getPonto(), examples[k].getPonto())
                        else:
                            dqj.append(calculaDistanciaEuclidiana(examples[j].getPonto(), examples[k].getPonto()))
                    apj /= nExemplos
                    if dqj:
                        bpj = min(dqj)
                        sj = (bpj - apj) / max(apj, bpj)
                        sorted_vals = sorted(membership_matrix[:, j], reverse=True)
                        upj, uqj = sorted_vals[:2]
                        numerador += ((upj - uqj) ** alpha) * sj
                        denominador += (upj - uqj) ** alpha
            fs = numerador / denominador if denominador != 0 else 0
            silhuetas.append(fs)
        return silhuetas

    @staticmethod
    def getFirstAndSecondBiggerPertinence(valores: np.ndarray):
        """Retorna a maior e a segunda maior pertinência"""
        lista = sorted(valores, reverse=True)
        return lista[0], lista[1]

    @staticmethod
    def calculaTipicidade(membership_matrix: np.ndarray):
        n, k = membership_matrix.shape
        typicality = np.zeros((n, k))
        for i in range(n):
            max_u_i = np.max(membership_matrix[i])
            for j in range(k):
                typicality[i][j] = membership_matrix[i][j] / max_u_i if max_u_i > 0 else 0
        return typicality

    @staticmethod
    def separateExamplesByClusterClassifiedByFuzzyCMeans(
        exemplos: List[Example],
        cntr: np.ndarray,
        u: np.ndarray,
        rotulo: float,
        alpha: float,
        theta: float,
        minWeight: int,
        t: int
    ) -> List[SPFMiC]:
        """Espelhando o método Java: cria SPFMiCs a partir do resultado do fuzzyCMeans"""
        matrizTipicidade = FuzzyFunctions.calculaTipicidade(u.T)
        sfMiCS = []

        for j, centroide in enumerate(cntr):
            spfmic = None
            SSDe, Me, Te = 0, 0, 0
            CF1pertinencias = np.zeros_like(centroide)
            CF1tipicidades = np.zeros_like(centroide)
            nClusterPoints = 0

            for k, ex in enumerate(exemplos):
                indiceMaior = np.argmax(u[:, k])
                if indiceMaior == j:
                    valorPert = u[j, k]
                    valorTip = matrizTipicidade[k][j]
                    ponto = ex.getPonto()
                    nClusterPoints += 1

                    if spfmic is None:
                        spfmic = SPFMiC(centroide, nClusterPoints, alpha, theta, t)
                        spfmic.setRotulo(rotulo)

                    dist = calculaDistanciaEuclidiana(centroide, ponto)
                    for i in range(len(ponto)):
                        CF1pertinencias[i] += ponto[i] * valorPert
                        CF1tipicidades[i] += ponto[i] * valorTip
                    Me += valorPert ** alpha
                    Te += valorTip ** theta
                    SSDe += valorPert * (dist ** 2)

            if spfmic and nClusterPoints >= minWeight:
                spfmic.setSSDe(SSDe)
                spfmic.setMe(Me)
                spfmic.setTe(Te)
                spfmic.setCF1pertinencias(CF1pertinencias)
                spfmic.setCF1tipicidades(CF1tipicidades)
                sfMiCS.append(spfmic)

        return sfMiCS

    @staticmethod
    def newSeparateExamplesByClusterClassifiedByFuzzyCMeans(
        exemplos: List[Example],
        cntr: np.ndarray,
        u: np.ndarray,
        rotulo: float,
        alpha: float,
        theta: float,
        minWeight: int,
        t: int
    ) -> List[SPFMiC]:
        """Espelhando o método Java: versão simplificada do agrupamento"""
        sfMiCS = []

        for j, centroide in enumerate(cntr):
            spfmic = None
            SSD = 0
            nClusterPoints = 0

            for k, ex in enumerate(exemplos):
                indiceMaior = np.argmax(u[:, k])
                if indiceMaior == j:
                    valorPert = u[j, k]
                    ponto = ex.getPonto()
                    nClusterPoints += 1

                    if spfmic is None:
                        spfmic = SPFMiC(centroide, nClusterPoints, alpha, theta, t)
                        spfmic.setRotulo(rotulo)

                    dist = calculaDistanciaEuclidiana(centroide, ponto)
                    SSD += valorPert * (dist ** 2)

            if spfmic and nClusterPoints >= minWeight:
                spfmic.setSSDe(SSD)
                sfMiCS.append(spfmic)

        return sfMiCS

    @staticmethod
    def getIndiceDoMaiorValor(array: np.ndarray) -> int:
        """Retorna o índice do maior valor < 1 (como no Java)"""
        index = 0
        maior = -1e9
        for i, val in enumerate(array):
            if val > maior and val < 1:
                index = i
                maior = val
        return index

    @staticmethod
    def separateByClasses(chunk: List[Example]) -> Dict[float, List[Example]]:
        examplesByClass = {}
        for ex in chunk:
            r = ex.getRotuloVerdadeiro()
            if r not in examplesByClass:
                examplesByClass[r] = []
            examplesByClass[r].append(ex)
        return examplesByClass
