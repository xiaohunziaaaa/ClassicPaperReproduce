from node.ops.op import Op
from optimizer import *
from initializer.initializer import *
import numpy as np
from node.tensor.tensor import Tensor


class Dense(Op):
    # parents: if, w
    # child: output_feature
    def __init__(self, units, name, optimizer, kernel_initializer='normal', dtype=np.float):
        self.trainable = True
        super(Dense, self).__init__(self.trainable, name)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.dtype = dtype
        self.optimizer = optimizer
        self.error = None
        self.pre_error = None

    def tpara(self, feature_shape):
        assert feature_shape[1] > 0
        funits = feature_shape[1]
        para_shape = (funits, self.units)
        grad_shape = (feature_shape[0], funits, self.units)
        tpara_tensor = Tensor(shape=para_shape, name='tensor/weights', initializer=self.kernel_initializer)
        self.grad = Tensor(shape=grad_shape, initializer='zeros')
        return tpara_tensor, tpara_tensor.name

    def output(self):
        # add output feature child
        of = Tensor((self.parents[0].value.shape[0], self.units), dtype=self.dtype)
        return of


    def forward(self):
        self.children[0].value = self.parents[0].value @ self.tparas['tensor/weights'].value
        return self.children[0]

    def backward(self):
        self.error = self.children[0].children[0].pre_error
        # cal errors
        # ATTENTION! Cal errors before update weights
        self.pre_error = Tensor(shape=self.parents[0].value.shape, initializer='zeros')
        self.pre_error.value = self.cal_pre_error(error_np=self.error.value,
                                                  tparas_np=self.tparas['tensor/weights'].value,
                                                  shape=self.parents[0].value.shape)

        # cal grad
        self.grad.value = self.cal_grad(error_np=self.error.value, parent_np=self.parents[0].value,
                                        grad_shape=self.grad.value.shape)

        # updatae grad
        self.tparas['tensor/weights'].value = self.update_grad(tparas_np=self.tparas['tensor/weights'].value,
                                                               grad_np=self.grad.value)

        return self.tparas['tensor/weights'].value

    # calculate grad with respect to each element in W
    def cal_grad(self, error_np, parent_np, grad_shape):
        # This is written for testing
        # self.error = Tensor(shape=(3, 2), initializer='ones')
        grad_np = np.zeros(shape=grad_shape)
        for i in range(parent_np.shape[0]):
            left = np.reshape(a=parent_np[i], newshape=(-1, 1))
            right = np.reshape(a=error_np[i], newshape=(-1, 1))
            grad_np[i] = left @ right.T
        return grad_np

    # most important method
    # calculate error of each batch
    def cal_pre_error(self, error_np, tparas_np, shape=None):
        # if previous op is trainable
        pre_error_np = np.zeros(shape=shape)
        for i in range(self.parents[0].value.shape[0]):
            left = tparas_np
            right = np.reshape(error_np[i], newshape=(-1, 1))
            temp = (left @ right).T
            pre_error_np[i] = np.reshape(a=temp, newshape=(-1))
        return pre_error_np

    def update_grad(self, tparas_np, grad_np):
        updated_grad = self.optimizer.update_grad(tparas_np, grad_np)
        return updated_grad

    def summary(self):
        print('Type=Node/Op/Dense, name={}, units={}'.format(self.name, self.units))


def test():
    test_parent = Tensor(shape=(2, 2), name='test_input', initializer='normal')

    from optimizer.SGD import SGD
    sgd = SGD(lr=0.001, name='SGD')
    dense = Dense(name='Flatten', units=2, optimizer=sgd, kernel_initializer='normal')
    # test init
    dense.summary()
    dense.parents.append(test_parent)
    # test output
    output = dense.output()
    output.summary()

    test_parent.summary()

    trainpara, tpname = dense.tpara(feature_shape=test_parent.value.shape)
    dense._addtpara(paranode=trainpara, name=tpname)
    # test forward
    dense.children.append(output)

    # testing backward
    dense.parents[0].value = np.asarray([[1, 2], [3, 4]])
    dense.tparas['tensor/weights'].value = np.asarray([[-0.1, -0.2], [0.3, 0.4]])
    error = np.asarray([[0.5, 0.4], [0.3, 0.2]])

    # cal errors
    dense.pre_error = Tensor(shape=dense.parents[0].value.shape, initializer='zeros')


    # cal grad
    dense.grad.value = dense.cal_grad(error_np=error, parent_np=dense.parents[0].value,
                                    grad_shape=dense.grad.value.shape)

    # updatae grad
    dense.tparas['tensor/weights'].value = dense.update_grad(tparas_np=dense.tparas['tensor/weights'].value,
                                                           grad_np=dense.grad.value)

    dense.pre_error.value = dense.cal_pre_error(error_np=error,
                                                tparas_np=dense.tparas['tensor/weights'].value,
                                                shape=dense.parents[0].value.shape)
    print(dense.pre_error.value)


if __name__=='__main__':
    test()



