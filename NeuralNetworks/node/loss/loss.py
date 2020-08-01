from node.node import Node

class Loss(Node):

    def __init__(self, name='unknown'):
        super(Loss, self).__init__(name=name)
        self.name = name
        self.type = 2
        self.pre_error = None

    def output(self):
        raise NotImplementedError('output should be implemented')

    def forward(self):
        raise NotImplementedError('forward should be implemented')


    def backward(self, preds, labels):
        raise NotImplementedError('backward should be implemented')

    def summary(self):
        raise NotImplementedError('summary should be implemented')