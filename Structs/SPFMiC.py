import numpy as np
from FuzzyFunctions.DistanceMeasures import calculaDistanciaEuclidiana
from Structs.Example import Example

class SPFMiC:
    def __init__(self, centroide: np.ndarray, N: int, alpha: float, theta: float, t: int):
        # No Java: CF1 começam zerados
        self.CF1pertinencias = np.zeros_like(centroide, dtype=float)
        self.CF1tipicidades = np.zeros_like(centroide, dtype=float)

        # No Java: Me e Te começam em 0
        self.Me = 1.0
        self.Te = 1.0
        self.SSDe = 0.0
        self.N = float(N)

        self.t = float(t)
        self.updated = float(t)
        self.created = float(t)

        self.rotulo = -1.0
        self.rotuloReal = -1.0
        self.centroide = np.array(centroide, dtype=float).copy()
        self.alpha = alpha
        self.theta = theta
        self.isObsolete = False
        self.isNull = False

    # ---------- Getters / Setters ----------
    def getCF1pertinencias(self):
        return self.CF1pertinencias

    def setCF1pertinencias(self, arr):
        self.CF1pertinencias = np.array(arr, dtype=float)

    def getCF1tipicidades(self):
        return self.CF1tipicidades

    def setCF1tipicidades(self, arr):
        self.CF1tipicidades = np.array(arr, dtype=float)

    def setSSDe(self, SSDe):
        self.SSDe = SSDe

    def getN(self):
        return self.N

    def getTheta(self):
        return self.theta

    def getT(self):
        return self.t

    def getRotulo(self):
        return self.rotulo

    def setRotulo(self, rotulo):
        self.rotulo = rotulo

    def getCentroide(self):
        return self.centroide

    def setCentroide(self, c):
        self.centroide = np.array(c, dtype=float)

    def setMe(self, me):
        self.Me = me

    def isNullFunc(self):
        return self.isNull

    def getRotuloReal(self):
        return self.rotuloReal

    def setRotuloReal(self, r):
        self.rotuloReal = r

    def getUpdated(self):
        return self.updated

    def setUpdated(self, u):
        self.updated = u

    def getCreated(self):
        return self.created

    def setTe(self, te):
        self.Te = te

    def setObsolete(self, b: bool):
        self.isObsolete = b

    def isObsoleteFunc(self):
        return self.isObsolete

    # ---------- Operações principais ----------
    def atualizaCentroide(self):
        nAtributos = len(self.CF1pertinencias)
        self.centroide = np.zeros(nAtributos)
        for i in range(nAtributos):
            self.centroide[i] = (
                (self.alpha * self.CF1pertinencias[i] + self.theta * self.CF1tipicidades[i])
                / (self.alpha * self.Te + self.theta * self.Me if (self.alpha * self.Te + self.theta * self.Me) != 0 else 1e-12)
            )

    def atribuiExemplo(self, exemplo: Example, pertinencia: float, tipicidade: float):
        dist = calculaDistanciaEuclidiana(exemplo.getPonto(), self.centroide)
        self.N += 1.0
        self.Me += pow(pertinencia, self.alpha)
        self.Te += pow(tipicidade, self.theta)
        self.SSDe += pow(dist, 2) * pertinencia
        for i in range(len(self.centroide)):
            self.CF1pertinencias[i] += exemplo.getPontoPorPosicao(i) * pertinencia
            self.CF1tipicidades[i] += exemplo.getPontoPorPosicao(i) * tipicidade
        self.atualizaCentroide()

    def calculaTipicidade(self, exemplo, n, K):
        gamma_i = self.getGamma(K)
        dist = calculaDistanciaEuclidiana(exemplo, self.centroide)

        if n <= 1 or gamma_i == 0:
            return 0.0

        expoente = 1.0 / (n - 1.0)
        return 1.0 / (1.0 + pow(((self.theta / (gamma_i if gamma_i != 0 else 1e-12)) * dist), expoente))

    def getGamma(self, K):
        return K * (self.SSDe / self.Me if self.Me != 0 else 1e-12)

    # ---------- Raio ----------
    def getRadiusWithWeight(self):
        if self.N == 0:
            return 0.0
        return np.sqrt(self.SSDe / self.N) * 2.0

    def getRadiusND(self):
        if self.N == 0:
            return 0.0
        return np.sqrt(self.SSDe / self.N)

    def getRadiusUnsupervised(self):
        if self.N == 0:
            return 0.0
        return np.sqrt(self.SSDe / self.N)

    # ---------- Utilidades ----------
    def toDoubleArray(self):
        return self.centroide

    def __repr__(self):
        return f"SPFMiC(rotulo={self.rotulo}, N={self.N}, centroide={self.centroide.tolist()})"
