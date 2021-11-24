import cupy as cp

class AdamOptimizer(object):
    def __init__(self, lr):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.lr = lr
        self.step = 0

    def update(self, input, grad):
        self.step += 1
        self.mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * (grad ** 2)
        mt_hat = self.mt / (1 - cp.power(self.beta1, self.step))
        vt_hat = self.vt / (1 - cp.power(self.beta2, self.step))
        output = input - self.lr * mt_hat / (cp.sqrt(vt_hat) + self.eps)
        return output

def init_optimizer(lr, optimizer):
    # stupid implementation here, need to find some way to amend this
    if not optimizer:
        if optimizer == 'Adam':
            self_optimizer = AdamOptimizer(lr)
    else:
        self_optimizer = False
    return lr, self_optimizer