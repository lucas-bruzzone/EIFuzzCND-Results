# ResultsForExample.py
# Tradução de EIFuzzCND.Evaluation.ResultsForExample.java :contentReference[oaicite:1]{index=1}

class ResultsForExample:
    def __init__(self, realClass: str, classifiedClass: str):
        self.realClass = realClass
        self.classifiedClass = classifiedClass

    def getRealClass(self) -> str:
        return self.realClass

    def setRealClass(self, realClass: str):
        self.realClass = realClass

    def getClassifiedClass(self) -> str:
        return self.classifiedClass

    def setClassifiedClass(self, classifiedClass: str):
        self.classifiedClass = classifiedClass

    def __repr__(self):
        return f"ResultsForExample(realClass={self.realClass}, classifiedClass={self.classifiedClass})"
