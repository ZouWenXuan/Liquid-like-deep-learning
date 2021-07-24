# =============================================================================
# Feedforward Neural Network for 10-class task
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
from model.Tools.Optimizers import Adam
import numpy as np

#%% model
class FNN:
    def __init__(self, N_in, N_hidden, N_out, bias=False):
        # Network Structure
        self.Nin = N_in
        self.Nhid = N_hidden
        self.Nout = N_out
        self.bias = bias
        
        # W
        self.w_hi = np.random.normal(0.0,1/np.sqrt(self.Nin),  [self.Nhid, self.Nin])
        self.w_hh = np.random.normal(0.0,1/np.sqrt(self.Nhid), [self.Nhid, self.Nhid])
        self.w_oh = np.random.normal(0.0,1/np.sqrt(self.Nhid), [self.Nout, self.Nhid])

        self.Adam_w_hi = Adam()    
        self.Adam_w_hh = Adam()   
        self.Adam_w_oh = Adam()  
        
        # b
        if bias:
            self.b_hi = np.random.normal(0.0, 1.0/np.sqrt(self.Nhid), [self.Nhid, 1])
            self.b_hh = np.random.normal(0.0, 1.0/np.sqrt(self.Nhid), [self.Nhid, 1])
            self.b_oh = np.random.normal(0.0, 1.0/np.sqrt(self.Nout), [self.Nout, 1])
          
            self.Adam_b_hi = Adam()    
            self.Adam_b_hh = Adam()   
            self.Adam_b_oh = Adam()  
        else:
            self.b_hi, self.b_hh, self.b_oh = 0, 0, 0
    
    def Relu(self,z):
        return np.maximum(0,z)
    
    def ReLuPrime(self,z):
        return np.where(z > 0, 1, 0)
    
    def orthogonal_matrix(self, shape):
        a = np.random.normal(0.0, 1.0, shape)
        u, _, v = np.linalg.svd(a, full_matrices=0)   
        q = u if u.shape == shape else v
        return q
    
    def orthogonal_initial(self):
        self.w_hi = self.orthogonal_matrix((self.Nhid, self.Nin))
        self.w_hh = self.orthogonal_matrix((self.Nhid, self.Nhid))
        self.w_oh = self.orthogonal_matrix((self.Nout, self.Nhid))      
                
        
    def backpropagate(self, x, y_, lr):
        # Feedforward
        b_size = x.shape[1]
        # I -> H1
        z_h1 = np.dot(self.w_hi, x) + self.b_hi 
        a_h1 = self.Relu(z_h1)
        # H1 -> H2
        z_h2 = np.dot(self.w_hh, a_h1) + self.b_hh
        a_h2 = self.Relu(z_h2)      
        # H2 -> O
        z_out = np.dot(self.w_oh, a_h2) + self.b_oh
        
        # softmax
        z_max = np.array(z_out.max(axis=0), ndmin=2)
        y_exp = np.exp(z_out-z_max) 
        sum_exp = np.array(y_exp.sum(axis=0), ndmin=2)
        y_softmax = y_exp/sum_exp
        error_CE=( -y_ * np.log(y_softmax+1e-40) ).sum(axis=0)
        self.error_CE = np.sum(error_CE)/np.size(error_CE)

        # backward
        # O -> H2
        delta_out = (y_softmax - y_)       
        gw_oh = np.dot(delta_out, a_h2.T)/b_size
        if self.bias:
            gb_oh = np.array(delta_out.mean(1), ndmin=2).T

        delta_hid2 = np.dot(self.w_oh.T, delta_out)*self.ReLuPrime(z_h2)
        gw_hh = np.dot(delta_hid2, a_h1.T)/b_size
        if self.bias:
            gb_hh = np.array(delta_hid2.mean(1), ndmin=2).T

        delta_hid1 = np.dot(self.w_hh.T, delta_hid2)*self.ReLuPrime(z_h1)
        gw_hi = np.dot(delta_hid1, x.T)/b_size
        if self.bias:
            gb_hi = np.array(delta_hid1.mean(1), ndmin=2).T
        
        # update
        self.w_hi = self.Adam_w_hi.optimize(lr, self.w_hi, gw_hi)
        self.w_hh = self.Adam_w_hh.optimize(lr, self.w_hh, gw_hh)
        self.w_oh = self.Adam_w_oh.optimize(lr, self.w_oh, gw_oh)       
        
        if self.bias:
            self.b_hi = self.Adam_b_hi.optimize(lr, self.b_hi, gb_hi)
            self.b_hh = self.Adam_b_hh.optimize(lr, self.b_hh, gb_hh)
            self.b_oh = self.Adam_b_oh.optimize(lr, self.b_oh, gb_oh)  
    
    # forward
    def forward(self, x):
        # Feedforward
        # I -> H1
        z_h1 = np.dot(self.w_hi, x) + self.b_hi 
        a_h1 = self.Relu(z_h1)
        # H1 -> H2
        z_h2 = np.dot(self.w_hh, a_h1) + self.b_hh
        a_h2 = self.Relu(z_h2)      
        # H2 -> O
        y = np.dot(self.w_oh, a_h2) + self.b_oh
        return y


#%% main: 
if __name__ == '__main__':
    def train(DNN, training_list, lr):
        x = (training_list[:,1:]).T
        y_list = (training_list[:,0]).astype(int)      
        y_ = (np.eye(10)[y_list]).T  
        DNN.backpropagate(x, y_, lr)
        train_error = DNN.error_CE
        return train_error
    
    def test(DNN, test_list):
        x = (test_list[:,1:]).T
        y_list = (test_list[:,0]).astype(int)      
        y_ = (np.eye(10)[y_list]).T  
        y = DNN.forward(x)
        # softmax
        y_max = np.array(y.max(axis=0), ndmin=2)
        y_exp = np.exp(y-y_max) 
        sum_exp = np.array(y_exp.sum(axis=0), ndmin=2)
        y_softmax = y_exp/sum_exp
        error_CE=( -y_ * np.log(y_softmax+1e-40) ).sum(axis=0)
        test_error = np.sum(error_CE)/np.size(error_CE)
        # accuracy
        choice = y.argmax(axis=0)
        test_accuracy = np.mean(y_list==choice)
        return test_error, test_accuracy
    
    
    #%%% load data
    data_path = "D:/PMI/Projects/LLDL/data/PCA_Mnist"
    train_data_all = np.loadtxt(data_path + "/train_data.txt")
    test_data_all = np.loadtxt(data_path + "/test_data.txt")
    
    #%%% hyper-parameter
    N_in = 20
    N_hidden = 15
    N_out = 10
    p = 2000
    lr = 0.1
    epochs = 1000
    
    train_data = train_data_all[0:p,0:(1+N_in)]
    test_data = test_data_all[:,0:(1+N_in)]
    
    #%%% manual test
    def train_weight(N_in, N_hidden, N_out, lr_initial):
        NN = FNN(N_in, N_hidden, N_out) 
        # NN.orthogonal_initial()
        train_error = np.zeros(epochs)
        for i in range(epochs):
            lr = lr_initial*(0.1**((i+1)//250000))
            train_error[i] = train(NN, train_data, lr)
            print('\rRunning: {:.2f}%'.format((i+1)/epochs*100), end='')
        test_error, test_accuracy = test(NN, test_data)
        return train_error, test_error, test_accuracy
    train_error, test_error, test_accuracy = train_weight(N_in, N_hidden, N_out, lr)
