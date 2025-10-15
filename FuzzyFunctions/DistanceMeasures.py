import numpy as np
from typing import List, Union
from Structs.Example import Example

def calculaDistanciaEuclidiana(ponto1: Union[Example, List[float], np.ndarray],
                               ponto2: Union[List[float], np.ndarray]) -> float:
    """
    Versão fiel ao Java:
    - Se ponto1 é Example: usa .getPonto()
    - Se ponto1 é array-like: usa diretamente
    - Sempre calcula distância Euclidiana por soma de quadrados + sqrt (sem np.linalg.norm)
    """
    if isinstance(ponto1, Example):
        a = np.array(ponto1.getPonto(), dtype=float)
    else:
        a = np.array(ponto1, dtype=float)

    b = np.array(ponto2, dtype=float)

    somatorio = 0.0
    for i in range(len(a)):
        somatorio += (a[i] - b[i]) ** 2

    return float(np.sqrt(somatorio))
