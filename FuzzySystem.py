import os
import pandas as pd
import numpy as np
from scipy.io import arff
from Phases.OfflinePhase import OfflinePhase
from Phases.OnlinePhase import OnlinePhase
from typing import List
import random

def main():
    random.seed(42)
    np.random.seed(42)
    dataset = "rbf"  # mesmo nome do Java
    caminho = os.path.join(os.getcwd(), "datasets", dataset, "")

    # parâmetros
    fuzzyfication: float = 2   # corrigido: mesmo nome que no Java
    alpha: float = 2
    theta: float = 1
    K: int = 4
    kshort: int = 4 #Número de clusters
    T: int = 40
    minWeightOffline: int = 0
    minWeightOnline: int = 15
    latencia: List[int] = [10000000]   # 2000, 5000, 10000, 10000000
    tChunk = 2000
    ts: int = 200
    phi: float = 0.2
    percentLabeled: List[float] = [1.0]

    # carrega dataset em ARFF (equivalente ao Java/Weka)
    train_path = os.path.join(caminho, dataset + "-train.arff")
    print(f"Tentando carregar: {train_path}")
    data_arff, meta = arff.loadarff(train_path)
    df = pd.DataFrame(data_arff)

    # separa atributos (X) e classe (y)
    X = df.iloc[:, :-1].astype(float).values
    # força conversão do rótulo nominal (bytes) -> string -> float
    y = pd.to_numeric(df.iloc[:, -1].astype(str), errors="coerce").astype(float).values

    # junta de volta no formato [features..., classValue]
    data = np.column_stack([X, y])
    print("Primeiras linhas processadas:")
    print(data[:5])  # debug: veja se agora está [f1, f2, ..., class]

    for lat in latencia:
        for labeled in percentLabeled:
            condicaoSatisfeita = False
            while not condicaoSatisfeita:
                # fase offline
                offlinePhase = OfflinePhase(dataset, caminho, fuzzyfication, alpha, theta, K, minWeightOffline)
                supervisedModel = offlinePhase.inicializar(data)

                # fase online
                onlinePhase = OnlinePhase(
                    caminho,
                    supervisedModel,
                    lat,
                    tChunk,
                    T,
                    kshort,
                    phi,
                    ts,
                    minWeightOnline,
                    labeled
                )
                onlinePhase.initialize(dataset)

                if onlinePhase.getTamConfusion() > 999:
                    # mesma lógica do Java: repete
                    continue
                else:
                    condicaoSatisfeita = True
                    break

if __name__ == "__main__":
    main()
