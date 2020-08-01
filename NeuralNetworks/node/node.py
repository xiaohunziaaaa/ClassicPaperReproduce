import numpy as np

class Node(object):
    def __init__(self, name='unknown', type=np.float):
        self.name = name
        self.type = type
        self.parents = []
        self.children = []
        return

    def _addparent(self, pnode):
        self.parents.append(pnode)
        return

    def _addchild(self, cnode):
        self.children.append(cnode)



