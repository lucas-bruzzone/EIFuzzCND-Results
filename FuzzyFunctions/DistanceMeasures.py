import numpy as np
from typing import List

def calculaDistanciaEuclidiana(ponto1, ponto2: List[float]) -> float:
    """
    Versão única (como no Java).
    Aceita:
      - ponto1 como lista/array de floats
      - ponto1 como Example (ou objeto com getPonto / ponto / get_point)
    E ponto2 sempre como lista/array de floats
    """
    if hasattr(ponto1, 'getPonto'):
        arr = ponto1.getPonto()
    elif hasattr(ponto1, 'ponto'):
        arr = ponto1.ponto
    elif hasattr(ponto1, 'get_point'):
        arr = ponto1.get_point()
    else:
        arr = ponto1

    a = np.array(arr, dtype=float)
    b = np.array(ponto2, dtype=float)
    return float(np.linalg.norm(a - b))
