# =============================================================================
# Optimizers
# =============================================================================

class Adam:
    '''
    Adam optimizer for gradient descent;
    Give a negative learning rate to achieve gradient accent.
    '''
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = eps
        self.m = 0
        self.s = 0
        self.t = 0
    
    def optimize(self, lr, origin, gradient):
        self.t += 1
        g = gradient
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.s = self.beta2*self.s + (1-self.beta2)*(g*g)
        self.mhat = self.m/(1-self.beta1**self.t)
        self.shat = self.s/(1-self.beta2**self.t)
        new = origin - lr*self.mhat/(pow(self.shat,0.5)+self.epislon)
        return new

class RMSprop:
    '''
    RMSprop optimizer for gradient descent;
    Give a negative learning rate to achieve gradient accent.
    '''
    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.epislon = eps
        self.s=0
        self.t=0
    
    def optimize(self, lr, origin, gradient):
        self.t += 1
        g = gradient
        self.s = self.beta*self.s + (1-self.beta)*(g*g)
        new = origin - self.lr*g/pow(self.s+self.epislon,0.5)
        return new
    
class Momentum:
    '''
    Momentum optimizer for gradient descent;
    Give a negative learning rate to achieve gradient accent.
    '''
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.v=0
    
    def optimize(self, lr, origin, gradient):
        g = gradient
        self.v = self.gamma*self.v + lr*g
        new = origin - self.v
        return new       
    
class SGD:
    '''
    SGD optimizer for gradient descent;
    Give a negative learning rate to achieve gradient accent.
    '''
    def __init__(self):
        pass
    
    def optimize(self, lr, origin, gradient):
        g = gradient
        new = origin - lr*g
        return new    


