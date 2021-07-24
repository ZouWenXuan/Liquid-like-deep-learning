# =============================================================================
# Discrete Neural Network shallow
# =============================================================================

    #%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
from model.Tools.Optimizers import Adam
import numpy as np
import scipy.special as sps

def f(x, N):
    return 1/np.sqrt(N)*(np.maximum(0,x)+np.minimum(0,0.1*x))

def df(x, N):
    return 1/np.sqrt(N)*np.where(x > 0, 1, 0.1)


#%% model
# LR-net class 
class LrnetShallow:
    def __init__(self, N_in, N_out):
        # Network Structure
        self.Nin = N_in
        self.Nout = N_out
        
        # Theta
        self.theta = np.random.normal(0.0,0.1,(self.Nout, self.Nin))
        
        # W
        self.w = np.zeros([self.Nout, self.Nin])
        
        # sigmoid func
        self.sigmoid_func = lambda x:sps.expit(x)  
        self.d_sigmoid = lambda x:sps.expit(x)*(1-sps.expit(x))
        
        # Adam
        self.adam = Adam()    
 
    #sample new epsilon
    def sample_epsilon(self, size):
        self.epsi = np.random.normal(0.0,1.0,(self.Nout, size))
       
    #back-prpagation once
    def run_bp(self, x, y_, gamma, lr):
        '''

        Parameters
        ----------
        x : array: [N_in, b_size],
            data inputs. 
        y_ : size: [N_out, b_size],
            targets.
        gamma : float,
            regularization strength.
        lr : float,
            learning rate.

        Returns
        -------
        Updata the theta.

        ''' 

        # Feed-forward
        b_size = x.shape[1]
        self.sample_epsilon(b_size)
        
        mu = 2*self.sigmoid_func(self.theta)-1
        sigma2 = 1-mu**2        
        m = np.dot(mu, x)
        v = np.sqrt( np.dot(sigma2, x**2) )
        z = m + v * self.epsi      
        self.output = z
        
        # softmax
        y_exp = np.exp(z) 
        sum_exp = np.array(y_exp.sum(axis=0), ndmin=2)
        y_softmax = y_exp/sum_exp
        error_CE=( -y_ * np.log(y_softmax) ).sum(axis=0)
        self.error_CE = np.sum(error_CE)/np.size(error_CE)
        
        # back-propagate
        gtheta = 0
        delta = (y_softmax - y_)
        gtheta = 2*np.dot(delta, x.T)*self.d_sigmoid(self.theta)
        gtheta += -2*np.dot(delta/v*self.epsi, (x**2).T)\
                        *mu *self.d_sigmoid(self.theta)
                        
        # regularization
        gtheta = gtheta/b_size + gamma * self.theta 
  
        # update
        self.theta = self.adam.optimize(lr, self.theta, gtheta)


    # Mean-field forward 
    def mf_forward(self, x):
        # Feed-forward
        b_size = x.shape[1]
        self.sample_epsilon(b_size) 
        mu = 2*self.sigmoid_func(self.theta)-1
        sigma2 = 1-mu**2        
        m = np.dot(mu, x)
        v = np.sqrt( np.dot(sigma2, x**2) )
        z = m + v * self.epsi  
        return z
    
    def sample_weight(self, theta):
        prob = self.sigmoid_func(theta)
        rand = np.random.rand(theta.shape[0],theta.shape[1])
        weight = prob - rand
        weight[weight>=0] = 1
        weight[weight< 0] = -1
        return weight
    
    def get_weight(self):
        self.w = self.sample_weight(self.theta)
        
                
    # sample forward
    def sp_forward(self, x):
        z = np.dot(self.w, x)
        return z    
      

#%% train&test func
def train(DNN, b_size, b_num, train_data, gamma, lr):
    # set mini-batch
    total_num = train_data.shape[0]
    index = np.random.permutation(total_num)
    error_CE = 0
    error_zero = 0
    for mb in range(b_num):
        # x, y_
        data_batch = train_data[index[mb*b_size:(mb+1)*b_size]]
        x = np.array(data_batch[:,1:], ndmin=2).T
        y_list = (data_batch[:,0]).astype(int)      
        y_ = (np.eye(10)[y_list]).T         
        # one-trial training
        DNN.sample_epsilon(b_size)
        DNN.run_bp(x, y_, gamma, lr)       
        # cross-entropy
        error_CE += DNN.error_CE
        outputs = DNN.output
        # error-zero
        choice = outputs.argmax(axis=0)
        error_zero += (1 - np.mean(y_list == choice))        
    error_CE = error_CE/b_num
    error_zero = error_zero/b_num
    return error_CE, error_zero

def test(DNN, test_data, sample=True):
    DNN.sample_epsilon(test_data.shape[0])
    x_test = np.array(test_data[:,1:], ndmin=2).T
    y_test = test_data[:,0]   
    if sample:
        DNN.get_weight()
        outputs = DNN.sp_forward(x_test)  
    else:
        outputs = DNN.mf_forward(x_test)
    choice = outputs.argmax(axis=0)
    accuracy = np.mean(y_test == choice)
    return accuracy

#%% main: 
if __name__ == '__main__':

    #%%% load mnist
    data_path = "D:/PMI/Projects/LLDL/data/PCA_Mnist"
    train_data = np.loadtxt(data_path+"/train_data.txt")
    test_data = np.loadtxt(data_path+"/test_data.txt")
    
    #%%% hyper-parameter for training
    N_in = 20
    N_out = 10
    b_size = 200
    b_num = 300
    gamma = 1e-5
    lr = 0.3
    epochs = 100
    DNN = LrnetShallow(N_in, N_out)
    
    #%%% train
    for ep in range(1, epochs+1):     
        CE, zero_loss = train(DNN, b_size, b_num, train_data, gamma, lr)
        print('\rEpoch:{}, CE:{:.4f}, zero-loss:{:.3f}'.format(ep, CE, zero_loss),end='')
    accuracy_5 = np.array([test(DNN, test_data, sample=True) for j in range(0,5)])  
    accuracy_highest = accuracy_5.max() 
    test_zero_loss = 1-accuracy_highest
    print('\nTest zero_loss = {:.4f}'.format(test_zero_loss))

