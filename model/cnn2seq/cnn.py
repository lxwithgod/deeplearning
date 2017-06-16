import numpy as np
import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from deeplearning.model.baseModel import BaseModel


class ConvNet(BaseModel):
    def __init__(self, hidden_state, dtype=np.float32):
        self.dtype = dtype
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        super(ConvNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 3, stride=1, pad=1, initialW=W)
            self.bn1 = L.BatchNormalization(size=96, dtype=np.float32)
            self.conv2 = L.Convolution2D(None, 96, 3, stride=1, pad=1, initialW=W)
            self.bn2 = L.BatchNormalization(size=96, dtype=dtype)
            self.conv3 = L.Convolution2D(None, 96, 3, stride=1, pad=1, initialW=W)
            self.bn3 = L.BatchNormalization(size=96, dtype=dtype)
            self.conv4 = L.Convolution2D(None, hidden_state, 3, stride=1, pad=1, initialW=W)
            self.bn4 = L.BatchNormalization(size=hidden_state, dtype=dtype)

    def __call__(self, xs):
        h = self.conv1(xs)
        h = F.relu(self.bn1(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv2(h)
        h = F.relu(self.bn2(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv3(h)
        h = F.relu(self.bn3(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv4(h)
        h = F.relu(self.bn4(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        #
        # h = self.conv5(h)
        # h = F.relu(self.bn5(h))
        # h = F.max_pooling_2d(h, 3, stride=2)

        batch_size, chanel, height, width = h.shape
        h = F.reshape(h, shape=(batch_size, chanel, height * width))

        seq = F.transpose(h, axes=(0, 2, 1))

        return seq


if __name__ == '__main__':
    predictor = ConvNet(10)

    img = np.ones(shape=(2, 3, 64, 64), dtype=np.float32)
    res1 = predictor(img)
    print(res1)
    # res1 = res1.data > 0
    # print(res1 + res1)
