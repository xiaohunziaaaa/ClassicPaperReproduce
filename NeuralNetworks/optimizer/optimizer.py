
class Optimizer(object):
    def __init__(self, name, lr=0.0001):
        self.name = name
        self.lr = lr

    def updata_grad(self):
        raise NotImplementedError('output method should be implemented')

    def summary(self):
        raise NotImplementedError('output method should be implemented')
