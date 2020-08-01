from node.node import Node
import numpy as np
class Tensor(Node):

    def __init__(self, shape, name='unknown', dtype=np.float, initializer='zeros'):
        super(Tensor, self).__init__(name, 1)
        if initializer == 'zeros':
            self.value = np.zeros(shape=shape, dtype=dtype)
        if initializer == 'normal':
            self.value = np.random.normal(size=shape)
        if initializer == 'ones':
            self.value = np.ones(shape=shape, dtype=dtype)
        self.type = 0
        return

    def summary(self):
        print('Type=Tensor: name={}, shape={}'.format(self.name, self.value.shape))
