import re


class MetricAccuracy:
    def __init__(self):
        self.answers = []
        self.predicts = []
        self.results = []

    def compare(self, answer, predict):
        self.answers.append(answer)
        self.predicts.append(predict)
        result = 1 if answer == predict else 0
        self.results.append(result)
        
    def get_result(self):
        return sum(self.results) / len(self.results)
    
    
class MetricInList:
    def __init__(self):
        self.answers = []
        self.predicts = []
        self.results = []

    def compare(self, answer, predict):
        self.answers.append(answer)
        self.predicts.append(predict)
        result = 1 if answer in predict else 0
        self.results.append(result)
        
    def get_result(self):
        return sum(self.results) / len(self.results)
    
    
class MetricInListForCombined(MetricInList):
    def compare(self, answer, predict):
        answer = self.normalize_regimen(answer)
        predict = [self.normalize_regimen(regimen) for regimen in predict]
        self.answers.append(answer)
        self.predicts.append(predict)
        result = 1 if answer in predict else 0
        self.results.append(result)
        
    def normalize_regimen(self, regimen):
        regimen = [re.sub(r'(?P<drug>\w+)\((?P<type>\w+)\)', r'\1', drug) for drug in regimen.split(',')]
        regimen = ','.join(sorted(regimen))
        return regimen