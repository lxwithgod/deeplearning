import numpy as np
import sys
from chainer import optimizers, serializers, Variable, cuda

from deeplearning.dataProcess.showInputLabels import showInputLabels
from deeplearning.dataProcess.yzm import YzmData
from deeplearning.model.baseModel import BaseModel
from deeplearning.model.cnn2seq.cnn import ConvNet

from deeplearning.model.cnn2seq.decoder import Decoder


class Cnn2TextSeq(BaseModel):
    def __init__(self, vocab_size=10, word_size=10, hidden_state=10, class_num=10):
        self.hidden_state = hidden_state
        super(Cnn2TextSeq, self).__init__()
        with self.init_scope():
            self.covnet = ConvNet(hidden_state=word_size)
            self.decoder = Decoder(vocab_size=vocab_size,
                                   word_size=word_size,
                                   hidden_state=hidden_state,
                                   class_num=class_num)

        self.loss = None
        self.acc = None

    def infer(self, inputs):
        hs = self.covnet(inputs)
        return self.decoder(hs)

    def __call__(self, inputs, targets):
        hs = self.covnet(inputs)
        loss, acc = self.decoder.loss_acc(hs, targets)
        self.loss = loss
        self.acc = acc
        return loss, acc


if __name__ == '__main__':

    yzmData = YzmData("../../resource/ddxl/info.txt")
    yzmData.load_classDict("./")

    model = Cnn2TextSeq(vocab_size=yzmData.classNum, word_size=128, hidden_state=128, class_num=yzmData.classNum)
    model.to_gpu(0)

    opt = optimizers.Adam()
    opt.setup(model)


    def loss(x, y):
        total_loss, acc = model(x, y)
        return total_loss


    # model.save_model("./model.npz",save_format="npz")
    model.load_model("./model.npz", load_format="npz")
    serializers.load_npz("./opt.npz", opt)

    step = 0
    while True:
        # x = Variable(data=np.zeros(shape=[10, 3, 60, 160], dtype=np.float32))
        # y = Variable(data=np.zeros(shape=[10, 5], dtype=np.int32))

        x, y = yzmData.nextBatch(testOrTrain="train", batch_size=64)

        x = Variable(x.astype(np.float32))
        y = Variable(y.astype(np.int32))
        x.to_gpu(0)
        y.to_gpu(0)
        opt.update(loss, x.data, y.data)

        print("step:%d\ttotal_loss:%f\terro_rate:%f" % (step, model.loss.data, model.acc.data))

        if step % 100 == 0:
            model.save_model("./model.npz", save_format="npz")
            serializers.save_npz("./opt.npz", opt)

            test_x, test_t = yzmData.nextBatch(testOrTrain="test", batch_size=40)
            test_x = Variable(test_x.astype(np.float32))
            test_t = Variable(test_t.astype(np.int32))
            test_x.to_gpu(0)
            test_t.to_gpu(0)

            infer_y = model.infer(test_x.data)
            print(infer_y)
            infer_y.to_cpu()
            test_x.to_cpu()
            showInputLabels(test_x.data, infer_y.data, yzmData.id2class)

            break
            # break
        step += 1
