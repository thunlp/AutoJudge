import pickle
import jieba



CasePath = './raw_data/case.bin'
LawPath = './raw_data/divorce_full.txt'


class Result(object):
    def __init__(self, result):
        self.plea = result[0]
        self.raw_result = result[1]
        self.result = result[2]  # binary
        self.extracted_law = result[3]


class DataInstance(object):
    def __init__(self, instance):
        self.fact = instance[0][0]
        self.pleas = instance[0][1]
        self.law = instance[0][2]
        self.raw_result = instance[3]
        self.result = [Result(Plea) for Plea in instance[4]]


class Reader(object):
    def __init__(self):
        self.law = {'divorce': [],
                    'child': [],
                    'maintenance': []}
        self.cases = []
        self._load()

    def _load(self):
        print("Start to load law text.")
        with open(LawPath, "r")as f:
            laws = f.readlines()
            laws = list(map(lambda x: list(
                filter(lambda x: len(x) > 0, list(jieba.cut(x.strip().replace(' ', ''), cut_all=False)))
            ), laws))
        print("law articles loaded.")
        self.law['divorce'].extend(laws[:26])
        self.law['child'].extend(laws[26:38])
        self.law['maintenance'].extend(laws[38:])

        with open(CasePath, "rb")as f:
            raw_data = pickle.load(f)
            self.cases.extend([DataInstance(instance) for instance in raw_data])
            print("cases loaded.")
