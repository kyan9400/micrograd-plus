import math

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {id(p): 0.0 for p in self.parameters}
        self.v = {id(p): 0.0 for p in self.parameters}

    def step(self):
        self.t += 1
        for p in self.parameters:
            gid = id(p)
            g = p.grad

            # Update biased first moment estimate
            self.m[gid] = self.beta1 * self.m[gid] + (1 - self.beta1) * g

            # Update biased second raw moment estimate
            self.v[gid] = self.beta2 * self.v[gid] + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected first and second moment
            m_hat = self.m[gid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[gid] / (1 - self.beta2 ** self.t)

            # Update parameter
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0
