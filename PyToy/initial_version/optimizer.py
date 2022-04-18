import cupy as cp

class AdamOptimizer(object):
    def __init__(self, lr):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.lr = lr
        self.step = 0

    def update(self, input, grad):
        if self.step == 0:
            self.mt = (1 - self.beta1) * grad
            self.vt = (1 - self.beta2) * cp.power(grad, 2)
        else:
            self.mt = cp.add(self.beta1 * self.mt, (1 - self.beta1) * grad)
            self.vt = cp.add(self.beta2 * self.vt, (1 - self.beta2) * cp.power(grad, 2))

        self.step += 1
        mt_hat = cp.divide(self.mt, (1 - cp.power(self.beta1, self.step)))
        vt_hat = cp.divide(self.vt, (1 - cp.power(self.beta2, self.step)))
        output = cp.subtract(input, cp.divide(self.lr * mt_hat, (cp.sqrt(vt_hat) + self.eps)))
        return output

def init_optimizer(lr, optimizer):
    # stupid implementation here, need to find some way to amend this
    if optimizer:
        if optimizer == 'Adam':
            self_optimizer = AdamOptimizer(lr)
    else:
        self_optimizer = False
    return self_optimizer