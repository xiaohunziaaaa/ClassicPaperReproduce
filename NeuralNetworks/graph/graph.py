from optimizer.SGD import SGD
from node.loss.cewithsoftmax import CEwithSoftmax

class Graph(object):
    def __init__(self, labels=10, lr=0.001, opitimizer_name='SGD', loss_name='CEwithSoftmax'):
        self.nodes = 0
        self.tensors = 0
        self.ops = 0
        self.head = None
        self.trial = None
        self.labels = labels
        self.loss_name = loss_name
        self.lr = lr
        self.optimizer_name = opitimizer_name


    def connect(self, node1, node2):
        # In all situation
        # node1 is a tensor(numpy.array in this implementation)
        # node2 is a op
        node2._addparent(pnode=node1)
        if node2.trainable:
            trainpara, tpname = node2.tpara(feature_shape=node1.value.shape)
            node2._addtpara(paranode=trainpara, name=tpname)

        output = node2.output()
        node1._addchild(cnode=node2)

        node2._addchild(cnode=output)
        output._addparent(pnode=node2)

        self.nodes += 1
        self.tensors += 1
        self.ops += 1
        self.trial = output

        return output

    def set_input(self, input):
        self.head = input
        self.trial = input
        self.tensors += 1
        self.nodes += 1
        return


    def set_loss(self):
        # By default, loss = cross entropy with softmax
        loss = CEwithSoftmax(name=self.loss_name, units=self.labels)
        loss._addparent(pnode=self.trial)
        self.loss = loss
        preds = loss.output()
        preds._addparent(pnode=loss)
        loss._addchild(cnode=preds)
        self.trial._addchild(cnode=loss)

        self.nodes += 1
        self.tensors += 1
        self.ops += 1
        self.trial = preds

    def set_optimizer(self, optimizer='SGD', lr=0.001):
        self.optimizer = SGD(lr=self.lr, name=self.optimizer_name)
        return


    # need to be changed when it comes to resnet-like network
    # this demo only support cascade network
    # the forward between layers controlled by graph but not layers themselves
    def forward(self, verbose=False):
        cur = self.head
        while (len(cur.children) > 0):
            # tensor node
            if cur.type == 0:
                cur = cur.children[0]
                continue
            # op node
            if cur.type == 1:
                temp = cur.forward()
                if verbose:
                    print(temp[0])
                cur = cur.children[0]
                continue
            # loss node
            if cur.type == 2:
                temp = cur.forward()
                if verbose:
                    print(temp[0])
                cur = cur.children[0]
                continue
        return

    def backward(self):
        cur = self.trial
        while (len(cur.parents)>0):
            if cur.type == 0:
                cur = cur.parents[0]
                continue
            # op node
            if cur.type == 1:
                cur.backward()
                cur = cur.parents[0]
                continue
            # loss node
            if cur.type == 2:
                cur.backward()
                cur = cur.parents[0]
                continue
        return

    def bprop(self, x_batch, y_batch):
        # set inputs and labels
        self.head.value = x_batch
        self.loss.labels = y_batch
        # step 1. forward
        self.forward()
        # step 2. backward
        self.backward()

    def summary(self):
        print('Type=Graph, ops(without loss)={}, tensors={}\n'.format(self.ops, self.tensors))
        print('The architecture is ')
        cur = self.head
        layers = 0
        while (len(cur.children) > 0):
            # tensor node
            if cur.type != 0:
                layers += 1
                print('Layer No.{} -------->'.format(layers),end='')
                cur.summary()
            else:
                print('Tensor shape ={}'.format(cur.value.shape))
            cur = cur.children[0]
        print('Prediction shape ={}'.format(cur.value.shape))
        print('The Optimizer is ')
        self.optimizer.summary()
        return


    def eval(self, x_test):
        self.head.value = x_test
        self.forward()
        return self.trial.value

    def getprebyname(self):
        pass


def test():
    from node.tensor.tensor import Tensor
    from node.ops.flatten import Flatten
    from node.ops.dense import Dense
    from node.ops.activation import Activation
    input = Tensor(shape=(10, 28, 28, 1), name='input')
    graph = Graph()
    graph.set_input(input)
    graph.set_optimizer()

    flatten = graph.connect(input, Flatten())
    hidden = graph.connect(flatten, Dense(units=128, optimizer=graph.optimizer, name='Dense1'))
    hidden_sigmoid = graph.connect(hidden, Activation(name='sigmoid', act_type='sigmoid'))
    output = graph.connect(hidden_sigmoid, Dense(units=10, optimizer=graph.optimizer, name='Dense2'))
    graph.set_loss()
    graph.summary()

if __name__=='__main__':
    test()

