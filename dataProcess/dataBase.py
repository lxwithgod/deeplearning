# coding:utf-8
import json
import random
import numpy as np


def store(data, jsonPath):
    with open(jsonPath, 'w') as json_file:
        json_file.write(json.dumps(data))


def load(jsonPath):
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        return data


class DataBase(object):
    def __init__(self, trainInfoPath=None, BE=True, trainDataRatio=0.7):
        self.id2class = {}
        self.class2id = {}
        self.classNum = None

        self.info = None
        self.trainData = None
        self.testData = None

        self.testIndex = 0
        self.trainIndex = 0

        if trainInfoPath is not None:
            self.info = self.open_info(trainInfoPath=trainInfoPath)
            wordset = self.get_class(self.info)
            self.id2class, self.class2id = self.get_dict(wordset, BE)
            self.classNum = len(self.id2class)
            self.sample(trainDataRatio=trainDataRatio)

    def sample(self, trainDataRatio):
        # random.shuffle(self.info)
        endIndex = int(trainDataRatio * len(self.info))
        self.trainData = self.info[:endIndex]
        self.testData = self.info[endIndex:]

    def do_line(self, line):
        raise "do_line is not implemented!"

    def open_info(self, trainInfoPath):
        with open(trainInfoPath, "r") as i:
            lines = i.readlines()

            info = map(self.do_line, lines)
            return list(info)

    def do_label(self, label):
        raise "do_label is not implemented!"

    def do_input(self, input):
        raise "do_input is not implemented!"

    def get_class(self, info):
        def getLabel(inputs):
            return inputs[1]

        wordset = set()
        labels = map(getLabel, info)
        for label in labels:
            for word in label:
                wordset.add(word)

        return wordset

    def get_dict(self, wordset, BE=True):
        assert isinstance(wordset, set)

        if BE:
            id2class = {k + 2: v for k, v in enumerate(wordset)}
            id2class[0] = "BOS"
            id2class[1] = "EOS"
        else:
            id2class = {k: v for k, v in enumerate(wordset)}

        class2id = {v: k for k, v in id2class.items()}

        return id2class, class2id

    def nextBatch(self, testOrTrain="test", batch_size=10):
        data = self.testData if testOrTrain == "test" else self.trainData
        index = self.testIndex if testOrTrain == "test" else self.trainIndex
        data_size = len(data)

        startIndex = batch_size * index % data_size
        endIndex = batch_size * (index + 1) % data_size

        if testOrTrain == "test":
            self.testIndex = index + 1
        else:
            self.trainIndex = index + 1
        if endIndex < startIndex:
            random.shuffle(self.trainData)
            startIndex = batch_size * (index + 1) % data_size
            endIndex = batch_size * (index + 2) % data_size
        someData = data[startIndex:endIndex]
        input = map(lambda inputs: self.do_input(inputs[0]), someData)
        label = map(lambda inputs: self.do_label(inputs[1]), someData)
        x, y = list(input), list(label)
        return np.array(x), np.array(y)


if __name__ == '__main__':
    dataSet = DataBase(u"/home/lx/文档/ddxl2/info.txt")
    print(dataSet.get_dict(dataSet.get_class(dataSet.info)))
