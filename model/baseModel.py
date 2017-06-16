import chainer
import cupy
import sys
from chainer import cuda, serializer, serializers, Variable
from chainer.dataset import to_device
import numpy as np


class BaseModel(chainer.Chain):
    def save_model(self, filename, save_format="hdf5"):
        gpu_flag = False
        if not self._cpu:
            self.to_cpu()
            gpu_flag = True

        getattr(serializers, 'save_{}'.format(save_format))(filename, self)
        if gpu_flag:
            self.to_gpu()

    def load_model(self, filename, load_format="hdf5"):
        getattr(serializers, 'load_{}'.format(load_format))(filename, self)

    def get_xp(self, array):
        if isinstance(array, Variable):
            array = array.data

        with cuda.get_device_from_array(array) as dev:
            xp = np if int(dev) == -1 else cuda.cupy
            return xp


if __name__ == '__main__':
    import chainer.links as L

    L.Classifier
