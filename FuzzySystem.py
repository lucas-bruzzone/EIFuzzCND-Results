# FuzzySystem.py
# Tradução espelhada de EIFuzzCND.FuzzySystem.java

import os
import pandas as pd
from Phases.OfflinePhase import OfflinePhase
from Phases.OnlinePhase import OnlinePhase

def main():
    dataset = "rbf"
    caminho = os.path.join(os.getcwd(), "datasets", dataset, "")

    # parâmetros
    fuzzification = 2
    alpha = 2
    theta = 1
    K = 4
    kshort = 4
    T = 40
    minWeightOffline = 0
    minWeightOnline = 15
    latencia = [10000000]   # pode incluir outros valores como no Java
    tChunk = 2000
    ts = 200
    phi = 0.2
    percentLabeled = [1.0]

    # carrega dataset em CSV (equivalente ao ARFF do Java)
    train_path = os.path.join(caminho, dataset + "-train.csv")
    print(f"Tentando carregar: {train_path}")
    data = pd.read_csv(train_path).values  # Agora já vira numpy.ndarray

    for lat in latencia:
        for labeled in percentLabeled:
            condicaoSatisfeita = False
            while not condicaoSatisfeita:
                # fase offline
                offlinePhase = OfflinePhase(dataset, caminho, fuzzification, alpha, theta, K, minWeightOffline)
                supervisedModel = offlinePhase.inicializar(data)  # passa o array, não o caminho

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
                    # no Java repetia a execução — aqui apenas logamos
                    print("Reexecutando para mesma latência...")
                else:
                    condicaoSatisfeita = True
                    break

if __name__ == "__main__":
    main()
