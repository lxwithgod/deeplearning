import numpy as np
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from deeplearning.model.baseModel import BaseModel
from deeplearning.model.cnn2seq.cnn import ConvNet


class GlobalAttention(BaseModel):
    def __init__(self, hidden_size=10, dtype=np.float32):
        self.dtype = dtype
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        self.hidden_size = hidden_size
        super(GlobalAttention, self).__init__(
            w1=L.Linear(hidden_size, hidden_size, initialW=W),
            w2=L.Linear(hidden_size, hidden_size, initialW=W)
        )

    def __call__(self, hs, s):

        batch_size = hs.shape[0]
        step = hs.shape[1]
        hidden_size = self.hidden_size
        weight_s = F.broadcast_to(F.expand_dims(self.w1(s), axis=1), shape=(batch_size, step, hidden_size))
        weight_a = [F.expand_dims(self.w2(hs[:, h, :]), axis=1) for h in range(step)]

        weight = F.concat(weight_a, axis=1) + weight_s
        weight = F.softmax(weight, axis=1)
        out = F.mean(weight * hs, axis=1)
        return out


if __name__ == '__main__':
    predictor = ConvNet(hidden_state=8)
    globalAttention = GlobalAttention(hidden_size=8)
    img = np.ones(shape=(2, 3, 64, 64), dtype=np.float32)
    s = np.ones(shape=(2, 8), dtype=np.float32)
    res1 = predictor(img)
    res2 = globalAttention(res1, s)
    print(res1.shape, res2.shape)
    print(res2)
