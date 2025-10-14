# Example.py (ajustes pequenos)
import numpy as np
from typing import List

class Example:
    def __init__(self, ponto, hasLabel: bool, tempo: int = 0):
        arr = np.array(ponto, dtype=float).ravel()  # <- FLATTEN SEMPRE

        if hasLabel:
            if arr.size < 2:
                raise ValueError(f"Exemplo rotulado precisa de pelo menos 1 feature + 1 rótulo. Recebi shape={arr.shape}")
            self.ponto = arr[:-1]
            self.rotuloVerdadeiro = float(arr[-1])
        else:
            self.ponto = arr
            self.rotuloVerdadeiro = -1.0

        self.tempo = tempo
        self.rotuloClassificado = -1.0

    def getPonto(self) -> List[float]:
        return self.ponto.tolist()

    def getPontoPorPosicao(self, i: int) -> float:
        return float(self.ponto[i])

    def getRotuloVerdadeiro(self) -> float:
        return float(self.rotuloVerdadeiro)

    def setRotuloClassificado(self, v: float):
        self.rotuloClassificado = float(v)

    def getRotuloClassificado(self) -> float:
        return float(self.rotuloClassificado)

    def getTime(self) -> int:
        return int(self.tempo)
