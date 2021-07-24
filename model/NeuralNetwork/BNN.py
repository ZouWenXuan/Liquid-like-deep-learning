# =============================================================================
# Binary Neural Network
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
class Lrnet:
    def __init__(self, N_in, N_hidden, N_out):
        # Network Structure
        self.Nin = N_in
        self.Nhid = N_hidden
        self.Nout = N_out
        
        # Theta
        self.theta_hi = np.random.normal(0.0,0.1,(self.Nhid, self.Nin))
        self.theta_hh = np.random.normal(0.0,0.1,(self.Nhid, self.Nhid))
        self.theta_oh = np.random.normal(0.0,0.1,(self.Nout, self.Nhid))
        
        # W
        self.w_hi = np.zeros([self.Nhid, self.Nin])
        self.w_hh = np.zeros([self.Nhid, self.Nhid])
        self.w_oh = np.zeros([self.Nout, self.Nhid])
        
        # sigmoid func
        self.sigmoid_func = lambda x:sps.expit(x)  
        self.d_sigmoid = lambda x:sps.expit(x)*(1-sps.expit(x))
        
        # Adam
        self.adam_hi = Adam()    
        self.adam_hh = Adam()   
        self.adam_oh = Adam()  
    
    
    #sample new epsilon
    def sample_epsilon(self, size):
        self.epsi_hid1 = np.random.normal(0.0,1.0,(self.Nhid, size))
        self.epsi_hid2 = np.random.normal(0.0,1.0,(self.Nhid, size))
        self.epsi_out = np.random.normal(0.0,1.0,(self.Nout, size))
       
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
        
        # hidden layer 1
        mu_hi = 2*self.sigmoid_func(self.theta_hi)-1
        sigma2_hi = 1-mu_hi**2        
        m_hid1 = np.dot(mu_hi, x)
        v_hid1 = np.sqrt( np.dot(sigma2_hi, x**2) )
        z_hid1 = m_hid1 + v_hid1 * self.epsi_hid1
        N_hid1 = x.shape[0]
        a_hid1 = f(z_hid1, N_hid1)

        # hidden layer 2
        mu_hh = 2*self.sigmoid_func(self.theta_hh)-1
        sigma2_hh = 1-mu_hh**2        
        m_hid2 = np.dot(mu_hh, a_hid1)
        v_hid2 = np.sqrt( np.dot(sigma2_hh, a_hid1**2) )
        z_hid2 = m_hid2 + v_hid2 * self.epsi_hid2
        N_hid2 = a_hid1.shape[0]
        a_hid2 = f(z_hid2, N_hid2)
        
        # output layer
        mu_oh = 2*self.sigmoid_func(self.theta_oh)-1
        sigma2_oh = 1-mu_oh**2 
        m_out = np.dot(mu_oh, a_hid2)
        v_out = np.sqrt( np.dot(sigma2_oh, a_hid2**2) )
        z_out = m_out + v_out * self.epsi_out        
        self.output = z_out*1
        
        # softmax
        y_exp = np.exp(z_out) 
        sum_exp = np.array(y_exp.sum(axis=0), ndmin=2)
        y_softmax = y_exp/sum_exp
        error_CE=( -y_ * np.log(y_softmax) ).sum(axis=0)
        self.error_CE = np.sum(error_CE)/np.size(error_CE)
        
        # back-propagate
        gtheta_oh, gtheta_hh, gtheta_hi = 0, 0, 0
        # output layer
        delta_out = (y_softmax - y_)
        gtheta_oh = 2*np.dot(delta_out, a_hid2.T)*self.d_sigmoid(self.theta_oh)
        gtheta_oh += -2*np.dot(delta_out/v_out*self.epsi_out, (a_hid2**2).T)\
                        *mu_oh *self.d_sigmoid(self.theta_oh)

        # hidden layer 2
        delta_hid2 = np.dot(mu_oh.T, delta_out)*df(z_hid2, N_hid2)
        delta_hid2 += a_hid2*np.dot(sigma2_oh.T, delta_out/v_out*self.epsi_out)\
                            *df(z_hid2, N_hid2)
        gtheta_hh = 2*np.dot(delta_hid2, a_hid1.T)*self.d_sigmoid(self.theta_hh)
        gtheta_hh += -2*np.dot(delta_hid2/v_hid2*self.epsi_hid2, (a_hid1**2).T)\
                        *mu_hh *self.d_sigmoid(self.theta_hh)

        # hidden layer 1
        delta_hid1 = np.dot(mu_hh.T, delta_hid2)*df(z_hid1, N_hid1)
        delta_hid1 += a_hid1*np.dot(sigma2_hh.T, delta_hid2/v_hid2*self.epsi_hid2)\
                            *df(z_hid1, N_hid1)
        gtheta_hi = 2*np.dot(delta_hid1, x.T)*self.d_sigmoid(self.theta_hi)
        gtheta_hi += -2*np.dot(delta_hid1/v_hid1*self.epsi_hid1, (x**2).T)\
                        *mu_hi *self.d_sigmoid(self.theta_hi)
           
        # regularization
        gtheta_hi = gtheta_hi/b_size + gamma * self.theta_hi 
        gtheta_hh = gtheta_hh/b_size + gamma * self.theta_hh 
        gtheta_oh = gtheta_oh/b_size + gamma * self.theta_oh 
  
        # update
        self.theta_hi = self.adam_hi.optimize(lr, self.theta_hi, gtheta_hi)
        self.theta_hh = self.adam_hh.optimize(lr, self.theta_hh, gtheta_hh)
        self.theta_oh = self.adam_oh.optimize(lr, self.theta_oh, gtheta_oh)
    

    # Mean-field forward 
    def mf_forward(self, x):
        # Feed-forward
        b_size = x.shape[1]
        self.sample_epsilon(b_size)
        
        # hidden layer 1
        mu_hi = 2*self.sigmoid_func(self.theta_hi)-1
        sigma2_hi = 1-mu_hi**2        
        m_hid1 = np.dot(mu_hi, x)
        v_hid1 = np.sqrt( np.dot(sigma2_hi, x**2) )
        z_hid1 = m_hid1 + v_hid1 * self.epsi_hid1
        N_hid1 = x.shape[0]
        a_hid1 = f(z_hid1, N_hid1)

        # hidden layer 2
        mu_hh = 2*self.sigmoid_func(self.theta_hh)-1
        sigma2_hh = 1-mu_hh**2        
        m_hid2 = np.dot(mu_hh, a_hid1)
        v_hid2 = np.sqrt( np.dot(sigma2_hh, a_hid1**2) )
        z_hid2 = m_hid2 + v_hid2 * self.epsi_hid2
        N_hid2 = a_hid1.shape[0]
        a_hid2 = f(z_hid2, N_hid2)
        
        # output layer
        mu_oh = 2*self.sigmoid_func(self.theta_oh)-1
        sigma2_oh = 1-mu_oh**2 
        m_out = np.dot(mu_oh, a_hid2)
        v_out = np.sqrt( np.dot(sigma2_oh, a_hid2**2) )
        z_out = m_out + v_out * self.epsi_out
        return z_out
    
    def sample_weight(self, theta):
        prob = self.sigmoid_func(theta)
        rand = np.random.rand(theta.shape[0],theta.shape[1])
        weight = prob - rand
        weight[weight>=0] = 1
        weight[weight< 0] = -1
        return weight
    
    def get_weight(self):
        self.w_hi = self.sample_weight(self.theta_hi)
        self.w_hh = self.sample_weight(self.theta_hh)
        self.w_oh = self.sample_weight(self.theta_oh)        
               
    # sample forward
    def sp_forward(self, x, hidden=False):
        # hidden layer 1
        z_hid1 = np.dot(self.w_hi, x)
        N_hid1 = x.shape[0]
        a_hid1 = f(z_hid1, N_hid1)
        # hidden layer 2
        z_hid2 = np.dot(self.w_hh, a_hid1)
        N_hid2 = a_hid1.shape[0]
        a_hid2 = f(z_hid2, N_hid2)        
        # output layer 
        z_out = np.dot(self.w_oh, a_hid2)
        if hidden:
            return z_hid1, z_hid2, z_out
        return z_out    
      
#%% main: 
if __name__ == '__main__':
    def train(DNN, b_size, b_num, train_data, gamma, lr):
        # set mini-batch
        total_num = train_data.shape[0]
        index = np.random.permutation(total_num)
        error_CE = 0
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
        error_CE = error_CE/b_num
        return error_CE
    
    def test(DNN, test_data, sample=True):
        DNN.sample_epsilon(test_data.shape[0])
        x = np.array(test_data[:,1:], ndmin=2).T
        y_list = (test_data[:,0]).astype(int)      
        y_ = (np.eye(10)[y_list]).T   
        if sample:
            DNN.get_weight()
            y = DNN.sp_forward(x)  
        else:
            y = DNN.mf_forward(x)
        # softmax
        y_exp = np.exp(y) 
        sum_exp = np.array(y_exp.sum(axis=0), ndmin=2)
        y_softmax = y_exp/sum_exp
        error_CE=( -y_ * np.log(y_softmax) ).sum(axis=0)
        test_error = np.sum(error_CE)/np.size(error_CE)
        # accuracy
        choice = y.argmax(axis=0)
        test_accuracy = np.mean(y_list == choice)
        return test_error, test_accuracy

    #%%% load mnist
    data_path = "D:/PMI/Projects/LLDL/data/PCA_Mnist"
    train_data = np.loadtxt(data_path+"/train_data.txt")[0:500]
    test_data = np.loadtxt(data_path+"/test_data.txt")
    
    #%%% hyper-parameter for training
    N_in = 20
    N_hidden = 15
    N_out = 10
    b_size = 200
    b_num = 300
    gamma = 1e-5
    lr = 0.3
    epochs = 200
    DNN = Lrnet(N_in, N_hidden, N_out)  
    
    #%%% manual test
    # for ep in range(1, epochs+1):     
    #     CE = train(DNN, b_size, b_num, train_data, gamma, lr)
    #     print('\rEpoch:{}, CE:{:.4f}'.format(ep, CE),end='')
    # error_accuracy5 = np.array([test(DNN, test_data, sample=True) for j in range(0,50)])  
    # accuracy_highest = error_accuracy5[:,1].max()
    # print('\nTest accuracy = {:.4f}'.format(accuracy_highest))

    import time
    time1 = time.time()
    for ep in range(1, epochs+1):     
        train(DNN, b_size, b_num, train_data, gamma, lr) 
        print("\r{}".format(ep), end='')
    test_error, test_accuracy = test(DNN, test_data)    
    print('')
    print(test_accuracy)
    time2 = time.time()
    print(time2-time1)