import numpy as np
from node.loss.loss import Loss
from node.tensor.tensor import Tensor
from nmath.nmath import softmax

class CEwithSoftmax(Loss):
    ouput_error = None
    preds = None

    def __init__(self, units, name, dtype=np.float):
        super(CEwithSoftmax, self).__init__(name=name)
        self.units = units
        self.dtype = dtype

    def output(self):
        # only 1 batch, this just create tensor to connect the graph
        pred_labels = Tensor(shape=(self.parents[0].value.shape[0], self.units), name='CEwithSoftmax_loss', dtype=self.dtype)
        return pred_labels

    def forward(self):
        preds = softmax(self.parents[0].value)
        self.children[0].value = preds
        return preds


    def backward(self):
        # with combination of softmax and cross_entropy, delta/error = prediction - real_label
        # ATTENTION! This function calculate output(final dense) error but not own error
        assert self.labels.shape[0] > 0
        pre_error_np = self.children[0].value - self.labels
        self.pre_error = Tensor(shape=pre_error_np.shape, initializer='zeros')
        self.pre_error.value = pre_error_np
        return self.pre_error

    def summary(self):
        print('Type=Node/Loss/CEwithSoftmax, name={}, units={}'.format(self.name, self.units))
        return

def test():
    test_parent = Tensor(shape=(18, 10), name='test_input')

    loss = CEwithSoftmax(units=10, name='CEwithSoftmax')
    # test init
    loss.summary()
    # test output
    output = loss.output()
    output.summary()
    # test forward
    loss.parents.append(test_parent)
    loss.children.append(output)
    print(loss.forward())
    # test backward
    test_labels = Tensor(shape=(18, 10), name='test_label')
    print(loss.backward(test_labels))

if __name__=='__main__':
    test()


