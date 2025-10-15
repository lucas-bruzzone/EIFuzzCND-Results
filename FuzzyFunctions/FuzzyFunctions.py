import numpy as np
import skfuzzy as fuzz
from typing import List, Dict
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC

class FuzzyFunctions:
    @staticmethod
    def fuzzyCMeans(examples: List[Example], K: int, fuzzification: float):
        data = np.array([ex.getPonto() for ex in examples]).T  # shape: features x samples
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c=K, m=fuzzification, error=1e-6, maxiter=1000, init=None
        )
        # Wrap em objeto que imita Java
        return cntr, u

    @staticmethod
    def getFirstAndSecondBiggerPertinence(valores: np.ndarray, j: int = 0):
        """
        Versão fiel ao Java:
        Retorna a maior e a segunda maior pertinência de um array.
        O parâmetro 'j' é mantido apenas para compatibilidade, mas não é usado.
        """
        lista = sorted(valores, reverse=True)
        if len(lista) < 2:
            return (lista[0], 0.0) if lista else (0.0, 0.0)
        return lista[0], lista[1]

    @staticmethod
    def fuzzySilhouette(clusterer: dict, exemplos: List[Example], alpha: float):
        u = clusterer["membership"]  # shape (c, n)
        nExemplos = u.shape[1]
        silhuetas = []

        for i in range(u.shape[0]):  # clusters
            numerador = denominador = 0.0
            for j in range(nExemplos):
                indexClasse = np.argmax(u[:, j])
                if indexClasse == i:
                    apj: float = 0.0
                    dqj = []
                    for k in range(nExemplos):
                        if np.argmax(u[:, k]) == indexClasse:
                            apj += calculaDistanciaEuclidiana(exemplos[j].getPonto(), exemplos[k].getPonto())
                        else:
                            dqj.append(calculaDistanciaEuclidiana(exemplos[j].getPonto(), exemplos[k].getPonto()))
                    apj /= nExemplos
                    if dqj:
                        bpj: float = min(dqj)
                        sj: float = (bpj - apj) / max(apj, bpj)

                        # Agora chamando a função auxiliar, como no Java
                        upj, uqj = FuzzyFunctions.getFirstAndSecondBiggerPertinence(u[:, j], j)

                        numerador += ((upj - uqj) ** alpha) * sj
                        denominador += (upj - uqj) ** alpha
            fs: float = numerador / denominador if denominador != 0 else 0.0
            silhuetas.append(fs)
        return silhuetas

    @staticmethod
    def getFirstAndSecondBiggerPertinence(valores: np.ndarray, j: int):
        """
        Versão fiel ao Java:
        Retorna a maior e a segunda maior pertinência de um array.
        """
        lista = sorted(valores, reverse=True)
        if len(lista) < 2:
            return (lista[0], 0.0) if lista else (0.0, 0.0)
        return lista[0], lista[1]

    @staticmethod
    def calculaTipicidade(membership_matrix: np.ndarray):
        # membership_matrix: shape (n, k) como no Java
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
        """
        Espelha a versão Java:
        - exemplos: lista de Example
        - cntr: centroides (K × atributos)
        - u: matriz de pertinência (K × nExemplos)
        - rotulo: classe associada
        - alpha, theta: parâmetros fuzzy
        - minWeight: peso mínimo para validar cluster
        - t: tempo inicial
        """
        # matriz de tipicidade (equivalente ao calculaTipicidade no Java)
        matrizTipicidade = FuzzyFunctions.calculaTipicidade(u.T)

        sfMiCS: List[SPFMiC] = []

        for j, centroide in enumerate(cntr):
            spfmic = None
            SSDe, Me, Te = 0.0, 0.0, 0.0
            CF1pertinencias = np.zeros_like(centroide)
            CF1tipicidades = np.zeros_like(centroide)
            nClusterPoints = 0

            for k, ex in enumerate(exemplos):
                indiceMaior = np.argmax(u[:, k])  # índice do cluster mais provável
                if indiceMaior == j:
                    valorPert = u[j, k]
                    valorTip = matrizTipicidade[k][j]
                    ponto = np.array(ex.getPonto(), dtype=float)
                    nClusterPoints += 1

                    if spfmic is None:
                        # cria SPFMiC inicial
                        spfmic = SPFMiC(centroide, nClusterPoints, alpha, theta, t)
                        spfmic.setRotulo(rotulo)

                    dist = calculaDistanciaEuclidiana(centroide, ponto)

                    # acumula valores
                    CF1pertinencias += ponto * valorPert
                    CF1tipicidades += ponto * valorTip
                    Me += valorPert ** alpha
                    Te += valorTip ** theta
                    SSDe += valorPert * (dist ** 2)

            # valida se cluster foi criado
            if spfmic is not None and nClusterPoints >= minWeight:
                # soma valores acumulados ao SPFMiC
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
            clusterer: dict,
            rotulo: float,
            alpha: float,
            theta: float,
            minWeight: int,
            t: int
    ) -> List[SPFMiC]:
        cntr, u = clusterer["centroids"], clusterer["membership"]
        sfMiCS = []

        for j, centroide in enumerate(cntr):
            spfmic = None
            SSD = 0.0
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

            if spfmic is not None:
                if spfmic.getN() >= minWeight:
                    spfmic.setSSDe(SSD)
                # mesmo se não atingir minWeight, ainda adiciona
                sfMiCS.append(spfmic)

        return sfMiCS

    @staticmethod
    def getIndiceDoMaiorValor(array: np.ndarray) -> int:
        index = 0
        maior = -1e9
        for i, val in enumerate(array):
            if val > maior and val < 1:
                index = i
                maior = val
        return index

    @staticmethod
    def separateByClasses(chunk: List[Example]) -> Dict[float, List[Example]]:
        examplesByClass: Dict[float, List[Example]] = {}
        for ex in chunk:
            r = ex.getRotuloVerdadeiro()
            if r not in examplesByClass:
                examplesByClass[r] = []
            examplesByClass[r].append(ex)
        return examplesByClass
