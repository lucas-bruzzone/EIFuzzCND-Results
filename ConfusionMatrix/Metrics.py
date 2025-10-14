class Metrics:
    def __init__(self, accuracy, precision, recall, f1, tempo, unkMem, unknownRate):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1Score = f1
        self.tempo = tempo
        self.unkMem = unkMem
        self.unknownRate = unknownRate

    def getAccuracy(self): return self.accuracy
    def getPrecision(self): return self.precision
    def getRecall(self): return self.recall
    def getF1Score(self): return self.f1Score
    def getTempo(self): return self.tempo
    def getUnkMem(self): return self.unkMem
    def getUnknownRate(self): return self.unknownRate