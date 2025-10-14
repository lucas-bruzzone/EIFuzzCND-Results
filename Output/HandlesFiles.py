# HandlesFiles.py
# Tradução fiel de EIFuzzCND.Output.HandlesFiles.java

import csv
import os
from typing import List
from Structs.Example import Example
from ConfusionMatrix.Metrics import Metrics
from Evaluation.ResultsForExample import ResultsForExample


class HandlesFiles:

    @staticmethod
    def salvaNovidades(novidades: List[float], arquivo: str, latencia: int, percentLabeled: float):
        base = os.path.join(os.getcwd(), "datasets", arquivo, "graphics_data")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"{arquivo}{latencia}-{percentLabeled}-EIFuzzCND-novelties.csv")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Linha", "Novidade"])
            for i, nov in enumerate(novidades):
                writer.writerow([i, nov])

    @staticmethod
    def salvaResultados(examples: List[Example], arquivo: str, latencia: int, percentLabeled: float):
        base = os.path.join(os.getcwd(), "datasets", arquivo, "graphics_data")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"{arquivo}{latencia}-{percentLabeled}-EIFuzzCND-results.csv")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Linha", "Rotulo Verdadeiro", "Rotulo Classificado"])
            for i, ex in enumerate(examples, 1):
                rotulo_class = "unknown" if ex.getRotuloClassificado() == -1 else ex.getRotuloClassificado()
                writer.writerow([i, ex.getRotuloVerdadeiro(), rotulo_class])

    @staticmethod
    def salvaMetrics(tempo: int, acuracia: float, precision: float, recall: float, f1Score: float,
                     dataset: str, latencia: int, percentLabeled: float, unkMen: float, unknownRate: float, append: bool):
        base = os.path.join(os.getcwd(), "datasets", dataset, "graphics_data")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"{dataset}{latencia}-{percentLabeled}-EIFuzzCND-acuracia.csv")

        mode = "a" if append else "w"
        with open(path, mode, newline="") as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(["Tempo", "Acurácia", "Precision", "Recall", "F1-Score", "unkMen", "unknownRate"])
            writer.writerow([tempo, acuracia, precision, recall, f1Score, unkMen, unknownRate])

    @staticmethod
    def loadResults(caminho: str, numAnalises: int):
        measures = []
        with open(caminho, "r") as f:
            reader = csv.reader(f)
            next(reader)  # pula header
            for _ in range(numAnalises):
                row = next(reader)
                temp2, temp3 = row[1], row[2]
                measures.append(ResultsForExample(temp2, temp3))
        return measures

    @staticmethod
    def loadNovelties(caminho: str, numAnalises: int) -> List[float]:
        measures = []
        with open(caminho, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for _ in range(numAnalises):
                row = next(reader)
                measures.append(float(row[1]))
        return measures

    @staticmethod
    def loadMetrics(caminho: str, numAnalises: int,
                    acuracias: List[float], precisoes: List[float],
                    recalls: List[float], f1Scores: List[float],
                    unkRs: List[float], unknownRates: List[float]):
        with open(caminho, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != ["Tempo", "Acurácia", "Precision", "Recall", "F1-Score", "unkMen", "unknownRate"]:
                raise ValueError("Formato de cabeçalho inválido no arquivo: " + caminho)
            for _ in range(numAnalises):
                row = next(reader)
                acuracias.append(float(row[1]) * 100)
                precisoes.append(float(row[2]) * 100)
                recalls.append(float(row[3]) * 100)
                f1Scores.append(float(row[4]) * 100)
                unkRs.append(float(row[5]))
                unknownRates.append(float(row[6]) * 100)
