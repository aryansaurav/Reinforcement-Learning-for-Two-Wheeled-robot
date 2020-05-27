import numpy as np
import math

class reward:

    def __init__(self, type):
        self.type = type    # reward types: 'sum', 'rbf', 'dist'
        self.value = 0      # reward current value
        if type == 'sum':    # parameters of the reward
            self.features = []
        elif type == 'rbf':
            self.pos = []
            self.width = 1
        elif type == 'step':
            self.pos = []
            self.width = 1

        self.input = []     # input of the reward (position, vel etc)

    def evaluate(self, x = np.array([]), u = np.array([])):
        if self.type=='rbf':
            x = x[0:self.pos.shape[0]]
            result = np.linalg.norm((x-self.pos)/self.width)
            return math.exp(-result)
        elif self.type=='dist':
            # result = np.linalg.norm(u)
            if u.shape[0] >0:
                result = math.exp(abs(u[0]))
            else:
                result = 0
            return result
        elif self.type == 'sum':
            result = 0
            for f in range(len(self.features)):
                result += self.theta[f] * self.features[f].evaluate(x,u)
            return result
        elif self.type == 'step':
            x = x[0:self.pos.shape[0]]
            result = np.linalg.norm((x - self.pos) / self.width)
            if result > self.width:
                result = 0
            else:
                result = 1
            return result

        else:
            print('evaluation of unknown reward type')


    def setparams(self, x, w = 1):
        if self.type == 'sum':
            self.features = x
            self.theta = w
        elif self.type == 'rbf':
            self.pos = x
            self.width = w
        elif self.type == 'step':
            self.pos = x
            self.width = w
        else:
            print("No parameters to set for this reward.")
