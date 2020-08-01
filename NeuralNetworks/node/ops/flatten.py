from node.ops.op import Op
from node.tensor.tensor import Tensor
import numpy as np

class Flatten(Op):

    def __init__(self, name='Flatten', dtype=np.float):
        super(Flatten, self).__init__(trainable=False, name=name)
        self.dtype = dtype


    def output(self):
        temp_shape = self.parents[0].value.shape
        rest = 1
        for i in range(1,len(temp_shape)):
            rest = rest * temp_shape[i]
        of = Tensor(shape=(self.parents[0].value.shape[0], rest), dtype=self.dtype)
        return of

    def forward(self):
        batch = self.parents[0].value.shape[0]
        self.children[0].value = np.reshape(a=self.parents[0].value, newshape=(batch, -1))
        return self.children[0].value

    def backward(self):
        # untrainable: error go through this layer
        self.pre_error = Tensor(shape=self.parents[0].value.shape, initializer='normal')
        self.pre_error.value = np.reshape(a=self.children[0].children[0].pre_error.value, newshape=self.parents[0].value.shape)
        #self.pre_error.value = np.reshape(a=np.asarray([[0,1],[1,0]]), newshape=self.parents[0].value.shape)


        return self.pre_error.value

    def cal_grad(self):
        pass

    def cal_error(self):
        pass

    def update_grad(self):
        pass

    def summary(self):
        print('Type=Node/Op/Flatten, name={}'.format(self.name))


def test():
    test_parent = Tensor(shape=(2, 2, 1, 1), name='test_input')

    flatten = Flatten(name='Flatten')
    # test init
    flatten.summary()
    flatten.parents.append(test_parent)
    # test output
    output = flatten.output()
    output.summary()
    # test forward
    flatten.children.append(output)
    print(flatten.forward())
    # test backward
    print(flatten.backward())

if __name__=='__main__':
    test()
