from node.ops.op import Op
import numpy as np
from node.tensor.tensor import Tensor
from nmath.nmath import *

class Activation(Op):
    def __init__(self, name='Activation', dtype=np.float, act_type='sigmoid'):
        super(Activation, self).__init__(trainable=False, name=name)
        self.dtype = dtype
        self.act_type = act_type
        self.error = None
        self.pre_error = None


    def output(self):
        of = Tensor(shape=self.parents[0].value.shape, dtype=self.dtype)
        return of

    def forward(self):
        self.children[0].value = activation(a=self.parents[0].value, act_type=self.act_type)
        return self.children[0].value

    def backward(self):
        self.error = self.children[0].children[0].pre_error
        # for test
        # self.error = Tensor(shape=self.children[0].value.shape, initializer='normal')
        self.pre_error = Tensor(shape=self.parents[0].value.shape, initializer='normal')
        act_dif = dif_activation(a=self.parents[0].value, act_type=self.act_type)
        self.pre_error.value = act_dif * self.error.value
        return self.pre_error.value

    def cal_grad(self):
        pass

    def cal_error(self):
        pass

    def update_grad(self):
        pass

    def summary(self):
        print('Type=Node/Op/{}, name={}'.format(self.act_type, self.name))

def test():
    test_parent = Tensor(shape=(2, 2), name='test_input')

    act = Activation(name='Sigmoid', act_type='sigmoid')
    # test init
    act.summary()
    act.parents.append(test_parent)
    # test output
    output = act.output()
    # test forward
    act.children.append(output)
    # print(act.forward())
    # test backward

    # print(act.backward())

if __name__=='__main__':
    test()