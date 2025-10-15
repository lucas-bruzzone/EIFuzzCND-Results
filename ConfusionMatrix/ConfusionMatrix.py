# ConfusionMatrix.py
import os
import csv
from collections import defaultdict
from typing import Dict, List

from ConfusionMatrix.Metrics import Metrics

class ConfusionMatrix:
    def __init__(self):
        self.matrix: Dict[float, Dict[float, int]] = {}
        self.lastMerge: Dict[float, float] = {}

    def addInstance(self, trueClass: float, predictedClass: float):
        if trueClass not in self.matrix:
            self._addClass(trueClass)
        if predictedClass not in self.matrix:
            self._addClass(predictedClass)

        count = self.matrix[trueClass].get(predictedClass, 0)
        self.matrix[trueClass][predictedClass] = count + 1

    def _addClass(self, classLabel: float):
        self.matrix[classLabel] = {}
        for otherClass in self.matrix.keys():
            self.matrix[classLabel][otherClass] = self.matrix[classLabel].get(otherClass, 0)
            self.matrix[otherClass][classLabel] = self.matrix[otherClass].get(classLabel, 0)

    def printMatrix(self):
        print("\nConfusion Matrix:")
        print("\t" + "\t".join(str(c) for c in self.matrix.keys()))
        for trueClass in self.matrix.keys():
            row = [str(self.matrix[trueClass].get(pred, 0)) for pred in self.matrix.keys()]
            print(f"{trueClass}\t" + "\t".join(row))

    def saveMatrix(self, dataset: str, latencia: int, percentLabeled: float):
        current = os.path.abspath(".")
        filePath = os.path.join(current, "datasets", dataset, "graphics_data",
                                f"{dataset}{latencia}-{percentLabeled}-matrix.csv")

        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        file_exists = os.path.isfile(filePath)

        with open(filePath, "a" if file_exists else "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # cabeçalho
            if not file_exists:
                writer.writerow(["Classes"] + list(self.matrix.keys()))

            # escreve matriz
            for trueClass in self.matrix.keys():
                row = [trueClass] + [self.matrix[trueClass].get(pred, 0) for pred in self.matrix.keys()]
                writer.writerow(row)

    def getClassesWithNonZeroCount(self) -> Dict[float, List[float]]:
        result: Dict[float, List[float]] = {}
        for trueClass in self.matrix.keys():
            if 0 <= trueClass < 100:
                predictedNonZero = []
                for predClass in self.matrix.keys():
                    if predClass > 100:
                        count = self.matrix[trueClass].get(predClass, 0)
                        if count > 0:
                            predictedNonZero.append(predClass)
                if predictedNonZero:
                    result[trueClass] = predictedNonZero
        return result

    def mergeClasses(self, labels: Dict[float, List[float]]):
        for srcLabel, destLabels in labels.items():
            if srcLabel not in self.matrix:
                continue
            row1 = self.matrix[srcLabel]

            for destLabel in destLabels:
                if destLabel in self.matrix and srcLabel != destLabel:
                    row2 = self.matrix[destLabel]

                    # soma linhas
                    for column, value2 in row2.items():
                        row1[column] = row1.get(column, 0) + value2

                    # remove linha de destino
                    self.matrix.pop(destLabel)

                    # soma colunas
                    for rowLabel, row in self.matrix.items():
                        if destLabel in row:
                            value2 = row.pop(destLabel)
                            row[srcLabel] = row.get(srcLabel, 0) + value2

                    self.lastMerge[srcLabel] = destLabel

        # aplicar merges pendentes
        for srcLabel, destLabel in self.lastMerge.items():
            if destLabel in self.matrix:
                self.mergeClasses({srcLabel: [destLabel]})

    def updateConfusionMatrix(self, trueLabel: float):
        # Remove 1 ocorrência do desconhecido (-1)
        if -1.0 in self.matrix.get(trueLabel, {}):
            self.matrix[trueLabel][-1.0] -= 1

    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        total = 0
        tp_total = 0

        precisions = []
        recalls = []

        for trueLabel, row in self.matrix.items():
            tp_cls = row.get(trueLabel, 0)
            fp_cls = sum(self.matrix[t].get(trueLabel, 0) for t in self.matrix if t != trueLabel)
            fn_cls = sum(row[p] for p in row if p != trueLabel)

            tp_total += tp_cls
            total += sum(row.values())

            prec = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0.0
            rec = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)

        accuracy = tp_total / total if total > 0 else 0.0
        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        unknownRate = (unkMem / exc) if exc > 0 else 0.0

        return Metrics(accuracy, precision, recall, f1, tempo, unkMem, unknownRate)

    def countUnknow(self) -> int:
        count = 0
        for row in self.matrix.values():
            count += row.get(-1.0, 0)
        return count

    def getNumberOfClasses(self) -> int:
        return len(self.matrix)
