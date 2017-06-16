import cupy
import numpy as np
import chainer.functions as F
from chainer import initializers, Variable
import chainer.links as L

from deeplearning.model.baseModel import BaseModel
from deeplearning.model.cnn2seq.attention import GlobalAttention
from deeplearning.model.cnn2seq.cnn import ConvNet


class Decoder(BaseModel):
    def __init__(self, vocab_size=10, word_size=10, hidden_state=10, class_num=10, dtype=np.float32, maxLength=10,
                 layer_num=1):

        super(Decoder, self).__init__()
        with self.init_scope():
            self.word_size = word_size
            self.beginToken = 0
            self.endToken = 1
            self.dtype = dtype
            self.maxLength = maxLength
            W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)

            self.embed = L.EmbedID(vocab_size, word_size)
            self.embed_proj = L.Linear(word_size, hidden_state, initialW=W)
            self.out = L.Linear(in_size=hidden_state, out_size=class_num, initialW=W)
            self.attention = GlobalAttention(hidden_size=hidden_state)

            self.layer_num = layer_num
            for layer_id in range(self.layer_num):
                setattr(self, "rnn_cell_" + str(layer_id), L.GRU(in_size=hidden_state, out_size=hidden_state))

                # self.mids.append(mid)

    def reset_state(self):
        for mid in self._children:
            assert isinstance(mid, str)
            if mid.startswith("rnn_cell"):
                getattr(self, mid).reset_state()


                # self.mid.reset_state()

    def step(self, hs, h, id):
        attention_out = self.attention(hs, h)
        embed_out = self.embed(id)
        embed_out = self.embed_proj(embed_out)
        next_h = attention_out + h + embed_out
        for layer_id in range(self.layer_num):
            next_h = getattr(self, "rnn_cell_" + str(layer_id))(next_h)
            next_h = F.Dropout(0.75)(next_h)

        out = self.out(next_h)
        return next_h, out

    def __lossAccuray(self, out, target):
        loss = F.softmax_cross_entropy(x=out, t=target)

        # print("----------")
        # print(F.argmax(out,1).data)
        # print(target)

        error_num = (1 - F.accuracy(out, target)) * target.size
        # error_num =(F.argmax(out,1)==target).sum()
        return loss, error_num

    def loss_acc(self, hs, targets):
        xp = self.get_xp(targets)

        batch_size, step_len = targets.shape
        h = xp.zeros(shape=(batch_size, self.word_size), dtype=self.dtype)

        total_loss = 0
        total_error_num = 0
        self.reset_state()

        decoder_inputs = targets[:, 0:-1]
        decoder_targets = targets[:, 1:]

        for i in range(step_len - 1):
            decoder_input = decoder_inputs[:, i]
            decoder_target = decoder_targets[:, i]

            h, out = self.step(hs, h, decoder_input)

            loss, error_num = self.__lossAccuray(out, decoder_target)
            total_loss += loss
            total_error_num += error_num

        return total_loss, total_error_num / batch_size / step_len

    def _infer(self, hs):
        batch_size = hs.shape[0]
        xp = self.get_xp(hs)
        h = xp.zeros(shape=(batch_size, self.word_size), dtype=self.dtype)
        id = xp.zeros(shape=batch_size, dtype=np.int32)

        infer_result = []
        self.reset_state()
        for step in range(self.maxLength):
            h, out = self.step(hs, h, id)
            id = F.argmax(x=out, axis=1)

            infer_result.append(F.reshape(id, [-1, 1]))
        infer_result = F.concat(infer_result, axis=1)
        return infer_result

    def __call__(self, hs):
        # if isinstance(hs, np.ndarray):
        #     hs = self.move_to_gpu(hs)
        return self._infer(hs)


if __name__ == '__main__':
    predictor = ConvNet(10)
    decoder = Decoder(vocab_size=10, word_size=10, hidden_state=10, class_num=10)

    predictor.load_model("mode.hdf5")
    decoder.load_model("model2.hdf5")
    img = np.ones(shape=(2, 3, 64, 64), dtype=np.float32)

    hs = predictor(img)
    print(hs)

    print(decoder(hs))
    decoder.save_model("model2.hdf5")
