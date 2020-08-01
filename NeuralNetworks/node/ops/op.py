from node.node import Node


class Op(Node):
    def __init__(self, trainable, name):
        super(Op, self).__init__(name, 1)
        self.tparas = {}
        self.trainable = trainable

    def _addtpara(self, paranode, name='unknown'):
        assert self.trainable == True
        self.tparas[name] = paranode

    def output(self):
        raise NotImplementedError('output method should be implemented')

    def forward(self):
        raise NotImplementedError('forward method should be implemented')

    def backward(self):
        raise NotImplementedError('backward method should be implemented')

    def cal_grad(self):
        raise NotImplementedError('cal_grad method should be implemented')

    def cal_error(self):
        raise NotImplementedError('cal_error method should be implemented')

    def update_grad(self):
        raise NotImplementedError('backward method should be implemented')

    def summary(self):
        raise NotImplementedError('summary method should be implemented')