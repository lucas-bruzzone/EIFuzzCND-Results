import numpy as np
import skfuzzy as fuzz
from typing import List, Dict
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC


class FuzzyClusterResult:
    """Wrapper para compatibilidade com Java FuzzyKMeansClusterer"""
    def __init__(self, centroids: np.ndarray, membership: np.ndarray, examples: List[Example]):
        self.centroids = centroids
        self.membership = membership
        self._examples = examples
        self._clusters_cache = None
    
    def getClusters(self):
        """Retorna lista de clusters com centroides e pontos"""
        if self._clusters_cache is not None:
            return self._clusters_cache
        
        clusters = []
        for i in range(self.centroids.shape[0]):
            indices = [j for j in range(self.membership.shape[1]) 
                      if np.argmax(self.membership[:, j]) == i]
            points = [self._examples[j] for j in indices]
            
            cluster = {
                'centroid': self.centroids[i],
                'points': points
            }
            clusters.append(cluster)
        
        self._clusters_cache = clusters
        return clusters


class FuzzyFunctions:
    @staticmethod
    def fuzzyCMeans(examples: List[Example], K: int, fuzzification: float):
        data = np.array([ex.getPonto() for ex in examples]).T
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c=K, m=fuzzification, error=1e-6, maxiter=1000, init=None
        )
        return FuzzyClusterResult(cntr, u, examples)

    @staticmethod
    def getFirstAndSecondBiggerPertinence(valores: np.ndarray, j: int = 0):
        lista = sorted(valores, reverse=True)
        if len(lista) < 2:
            return (lista[0], 0.0) if lista else (0.0, 0.0)
        return lista[0], lista[1]

    @staticmethod
    def fuzzySilhouette(clusterer: FuzzyClusterResult, exemplos: List[Example], alpha: float):
        u = clusterer.membership
        nExemplos = u.shape[1]
        silhuetas = []

        for i in range(u.shape[0]):
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
                        upj, uqj = FuzzyFunctions.getFirstAndSecondBiggerPertinence(u[:, j], j)
                        numerador += ((upj - uqj) ** alpha) * sj
                        denominador += (upj - uqj) ** alpha
            fs: float = numerador / denominador if denominador != 0 else 0.0
            silhuetas.append(fs)
        return silhuetas

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
        matrizTipicidade = FuzzyFunctions.calculaTipicidade(u.T)
        sfMiCS: List[SPFMiC] = []

        for j, centroide in enumerate(cntr):
            # PRIMEIRO: conta quantos pontos pertencem a este cluster
            indices_do_cluster = [k for k in range(len(exemplos)) if np.argmax(u[:, k]) == j]
            nClusterPoints = len(indices_do_cluster)
            
            if nClusterPoints == 0:
                continue
            
            spfmic = None
            SSDe, Me, Te = 0.0, 0.0, 0.0
            CF1pertinencias = np.zeros_like(centroide)
            CF1tipicidades = np.zeros_like(centroide)
            primeiro_ponto = True

            for k in indices_do_cluster:
                ex = exemplos[k]
                valorPert = u[j, k]
                valorTip = matrizTipicidade[k][j]
                ponto = np.array(ex.getPonto(), dtype=float)

                if spfmic is None:
                    # N = tamanho TOTAL do cluster (como no Java)
                    spfmic = SPFMiC(centroide, nClusterPoints, alpha, theta, t)
                    spfmic.setRotulo(rotulo)

                dist = calculaDistanciaEuclidiana(centroide, ponto)

                Me += valorPert ** alpha
                Te += valorTip ** theta
                SSDe += valorPert * (dist ** 2)

                if not primeiro_ponto:
                    CF1pertinencias += ponto * valorPert
                    CF1tipicidades += ponto * valorTip
                else:
                    primeiro_ponto = False

            if spfmic is not None and nClusterPoints >= minWeight:
                CF1pertinencias += spfmic.getCF1pertinencias()
                CF1tipicidades += spfmic.getCF1tipicidades()
                
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
            clusterer: FuzzyClusterResult,
            rotulo: float,
            alpha: float,
            theta: float,
            minWeight: int,
            t: int
    ) -> List[SPFMiC]:
        centroides_list = clusterer.getClusters()
        u = clusterer.membership
        sfMiCS = []

        for j, cluster_info in enumerate(centroides_list):
            spfmic = None
            SSD = 0.0
            examples = cluster_info['points']
            nClusterPoints = len(examples)

            if nClusterPoints > 0:
                spfmic = SPFMiC(cluster_info['centroid'], nClusterPoints, alpha, theta, t)
                spfmic.setRotulo(rotulo)

                for ex in examples:
                    indexExample = exemplos.index(ex)
                    valorPert = u[j, indexExample]
                    dist = calculaDistanciaEuclidiana(cluster_info['centroid'], ex.getPonto())
                    SSD += valorPert * (dist ** 2)

                if nClusterPoints >= minWeight:
                    spfmic.setSSDe(SSD)
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