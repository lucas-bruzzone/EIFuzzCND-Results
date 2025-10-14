import csv, os
from collections import defaultdict
from typing import Dict, List
from ConfusionMatrix.Metrics import Metrics

class ConfusionMatrix:
    def __init__(self):
        self.matrix: Dict[float, Dict[float, int]] = {}
        self.lastMerge: Dict[float, float] = {}

    def addInstance(self, trueClass: float, predictedClass: float):
        if trueClass not in self.matrix:
            self.addClass(trueClass)
        if predictedClass not in self.matrix:
            self.addClass(predictedClass)

        self.matrix[trueClass][predictedClass] += 1

    def addClass(self, classLabel: float):
        if classLabel not in self.matrix:
            self.matrix[classLabel] = {}
            # cria linha completa
            for other in list(self.matrix.keys()):
                self.matrix[classLabel][other] = self.matrix[classLabel].get(other, 0)
            # adiciona nova coluna em todas as outras linhas
            for other in self.matrix.keys():
                if other != classLabel:
                    self.matrix[other][classLabel] = 0

    def printMatrix(self):
        print("\nConfusion Matrix:")
        print("\t" + "\t".join(str(c) for c in self.matrix.keys()))
        for trueClass, row in self.matrix.items():
            print(str(trueClass) + "\t" + "\t".join(str(row.get(pred, 0)) for pred in self.matrix.keys()))

    def saveMatrix(self, dataset: str, latencia: int, percentLabeled: float):
        base = os.path.join(os.getcwd(), "datasets", dataset, "graphics_data")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"{dataset}{latencia}-{percentLabeled}-matrix.csv")

        file_exists = os.path.exists(path)
        with open(path, "a" if file_exists else "w", newline="") as f:
            writer = csv.writer(f)
            # Sempre escreve cabeçalho como no Java
            writer.writerow(["Classes"] + list(self.matrix.keys()))
            for trueClass, row in self.matrix.items():
                writer.writerow([trueClass] + [row.get(pred, 0) for pred in self.matrix.keys()])

    def getClassesWithNonZeroCount(self) -> Dict[float, List[float]]:
        result = {}
        for trueClass, row in self.matrix.items():
            if 0 <= trueClass < 100:
                nonZeroPreds = [pred for pred, count in row.items() if pred > 100 and count > 0]
                if nonZeroPreds:
                    result[trueClass] = nonZeroPreds
        return result

    def mergeClasses(self, labels: Dict[float, List[float]]):
        for srcLabel, destLabels in labels.items():
            if srcLabel not in self.matrix:
                continue
            row1 = self.matrix[srcLabel]
            for dest in destLabels:
                if dest in self.matrix and dest != srcLabel:
                    row2 = self.matrix[dest]
                    # soma linhas
                    for col, v2 in row2.items():
                        row1[col] = row1.get(col, 0) + v2
                    # remove linha dest
                    self.matrix.pop(dest, None)
                    # soma colunas
                    for row in self.matrix.values():
                        if dest in row:
                            v2 = row.pop(dest)
                            row[srcLabel] = row.get(srcLabel, 0) + v2
                    self.lastMerge[srcLabel] = dest
        for src, dest in list(self.lastMerge.items()):
            if dest in self.matrix:
                self.mergeClasses({src: [dest]})

    def updateConfusionMatrix(self, trueLabel: float):
        if trueLabel in self.matrix and -1.0 in self.matrix[trueLabel]:
            self.matrix[trueLabel][-1.0] = max(0, self.matrix[trueLabel][-1.0] - 1)

    def calculateMetrics(self, tempo: int, unkMem: float, exc: float) -> Metrics:
        tp = fp = fn = 0
        total = 0
        for trueLabel, row in self.matrix.items():
            for predLabel, count in row.items():
                total += count
                if trueLabel == predLabel:
                    tp += count
                else:
                    # espelha a lógica Java: soma como FP e FN
                    fp += count
                    fn += count

        tn = total - tp - fp - fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        unknownRate = (unkMem / exc) if exc > 0 else 0
        return Metrics(accuracy, precision, recall, f1, tempo, unkMem, unknownRate)

    def countUnknow(self) -> int:
        return sum(row.get(-1.0, 0) for row in self.matrix.values())

    def getNumberOfClasses(self) -> int:
        return len(self.matrix)
