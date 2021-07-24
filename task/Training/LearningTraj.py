# =============================================================================
# Learning Traj
# =============================================================================

#%% import module
import numpy as np
from model.NeuralNetwork.Lrnet import Lrnet

#%% functions
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
    if sample:
        DNN.get_weight()
        y = DNN.sp_forward(x)  
    else:
        y = DNN.mf_forward(x)
    # accuracy
    choice = y.argmax(axis=0)
    test_accuracy = np.mean(y_list == choice)
    return test_accuracy



#%%% load mnist
data_path = "D:/PMI/Projects/LLDL/data/PCA_Mnist"
train_data = np.loadtxt(data_path+"/train_data.txt")
test_data = np.loadtxt(data_path+"/test_data.txt")

#%% hyper-parameter for training
N_in = 20
N_hidden = 15
N_out = 10
b_size = 200
b_num = 300
gamma = 1e-5
lr = 0.5
epochs = 200


#%%% Test
accuracy_10 = np.zeros((20, epochs))
error_10 = np.zeros((20, epochs))
for i in range(20):
    DNN = Lrnet(N_in, N_hidden, N_out)  
    accuracy = np.array([test(DNN, test_data, sample=True) for j in range(0,50)]).max()  
    accuracy_10[i][0] = accuracy*1  
    for ep in range(1, epochs):     
        CE = train(DNN, b_size, b_num, train_data, gamma, lr)
        accuracy = np.array([test(DNN, test_data, sample=True) for j in range(0,50)]).max()  
        accuracy_10[i][ep] = accuracy*1  
        error_10[i][ep-1] = CE*1
    error_10[i][-1] = train(DNN, b_size, b_num, train_data, gamma, lr)
    
#%% save
np.savetxt("accuracy_up.txt", accuracy_10[10:])
np.savetxt("error_up.txt", error_10[10:])