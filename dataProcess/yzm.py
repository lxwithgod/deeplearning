# coding:utf-8
from deeplearning.dataProcess.dataBase import DataBase, store, load
import numpy as np
import cv2


class YzmData(DataBase):
    def __init__(self, trainInfoPath, BE=True, trainDataRatio=0.7):
        self.trainInfoPath = trainInfoPath
        self.rootPath = trainInfoPath[:trainInfoPath.rfind("/") + 1]

        super(YzmData, self).__init__(trainInfoPath=trainInfoPath, BE=BE, trainDataRatio=trainDataRatio)

    def do_line(self, line):
        input, label = line.split(",")[:2]
        return input, label[:label.find("/")]

    def do_label(self, label):
        a = list(map(lambda word: self.class2id[word], label))
        a.insert(0, 0)
        a.append(1)
        # b=list(map(lambda id: self.id2class[id], a))
        # print(b)
        return a

    def do_input(self, input):
        img = (cv2.imread(self.rootPath + input) - 128.0) / 128.0
        return np.transpose(img, axes=[2, 0, 1])

    def save_classDict(self, file_path):
        store(self.id2class, file_path + "/id2class.json")
        store(self.class2id, file_path + "/class2id.json")

    def load_classDict(self, file_path):
        id2class = load(file_path + "/id2class.json")
        class2id = load(file_path + "/class2id.json")

        self.id2class={int(k):v for k,v in id2class.items()}
        self.class2id={k:int(v) for k,v in class2id.items()}


if __name__ == '__main__':
    yzmData = YzmData("../resource/ddxl/info.txt")
    x, y = yzmData.nextBatch(testOrTrain="test")
