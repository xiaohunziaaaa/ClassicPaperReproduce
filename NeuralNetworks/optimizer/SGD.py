from optimizer.optimizer import Optimizer
from node.tensor.tensor import Tensor
import numpy as np
class SGD(Optimizer):
    def __init__(self, name,lr):
        super(SGD, self).__init__(name=name, lr=lr)


    def update_grad(self, para, grad):
        mean_grad = np.mean(a=grad, axis=0)
        updated_para = para - self.lr * mean_grad
        return updated_para

    def summary(self):
        print('Type=Optimizer/SGD, name={}, lr={}'.format(self.name, self.lr))
def test():
    para = Tensor(shape=(10, 10), initializer='ones')
    grad = Tensor(shape=(10, 10), initializer='ones')
    optimizer = SGD(name='SGD', lr=0.001)
    optimizer.summary()
    print(optimizer.update_grad(para, grad))

if __name__=='__main__':
    test()
