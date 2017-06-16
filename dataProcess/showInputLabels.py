import numpy as np
import cv2 as cv


def showInputLabels(imgs, labels,js):
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    batch_size, height, width, chanel = imgs.shape
    for i in range(batch_size):
        cv.imshow("img_" + str(i), imgs[i])
        label_id=labels[i]
        label_name=[js[id] for id in label_id]
        print("img_" + str(i), label_name)
        cv.waitKey(0)
