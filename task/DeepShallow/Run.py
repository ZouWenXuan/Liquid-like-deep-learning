# =============================================================================
# Lrnet: deep & shallow
# =============================================================================

#%% import module
import numpy as np
import time
from model.NeuralNetwork.Lrnet import Lrnet
from model.NeuralNetwork.LrnetShallow import LrnetShallow

#%% function: deep
def train(NN, b_size, b_num, train_data, gamma, lr):
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
        NN.sample_epsilon(b_size)
        NN.run_bp(x, y_, gamma, lr)       
        # cross-entropy
        error_CE += NN.error_CE  
    error_CE = error_CE/b_num
    return error_CE


def test(NN, test_data, sample=True):
    NN.sample_epsilon(test_data.shape[0])
    x = np.array(test_data[:,1:], ndmin=2).T
    y_list = (test_data[:,0]).astype(int)      
    y_ = (np.eye(10)[y_list]).T   
    if sample:
        NN.get_weight()
        y = NN.sp_forward(x)  
    else:
        y = NN.mf_forward(x)
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


def run_trial(NN, N_in, N_hidden, N_out, train_data, b_size, b_num, test_data,\
              epochs, gamma, lr_initial):
    train_error_traj = np.zeros(epochs)
    for i in range(epochs):
        train_error = train(NN, b_size, b_num, train_data, gamma, lr)
        train_error_traj[i] = train_error
        print("\rTrainig Progress: {:.2f}%".format((i+1)/epochs*100), end='')
    print("")
    [test_error, test_accuracy] = np.array([test(NN, test_data) for j in range(0,20)]).mean(0)  
    return train_error_traj, test_error, test_accuracy


#%% load data
data_path = "D:/PMI/Projects/LLDL/task/DoubleDescent/PCA"
train_data_all = np.load(data_path + "/trainPCA50.npy")
test_data_all = np.load(data_path + "/testPCA50.npy")


#%%% hyper-parameter
N_in = 20
N_out = 10
p = 60000
b_size = 200
b_num = 300
lr = 0.5
gamma = 1e-5
epochs = 200

train_data = train_data_all[0:p,0:(1+N_in)]
test_data = test_data_all[:,0:(1+N_in)]

SNN = LrnetShallow(N_in, N_out)

#%% main
# Deep network
time1 = time.time()
Deep_dict = {}
for N_hidden in range(1,51,1):
    Deep_dict[N_hidden] = {}
    train_error_10 = np.zeros((10, epochs))
    test_error_10 = np.zeros(10)
    test_accuracy_10 = np.zeros(10)
    for r in range(10):
        print("Hidden Nodes: {}, Repeat: {}.".format(N_hidden, r))
        DNN = Lrnet(N_in, N_hidden, N_out)
        train_error, test_error, test_accuracy = run_trial(DNN, N_in, N_hidden, N_out,\
                           train_data, b_size, b_num, test_data, epochs, gamma, lr)
        train_error_10[r] = train_error*1
        test_error_10[r] = test_error*1
        test_accuracy_10[r] = test_accuracy*1
    Deep_dict[N_hidden]['train_error'] = train_error_10*1
    Deep_dict[N_hidden]['test_error'] = test_error_10*1
    Deep_dict[N_hidden]['test_accuracy'] = test_accuracy_10*1
np.save("Deep_dict.npy", Deep_dict)
time2 = time.time()
print("Total time cost: {}".format(time2-time1))


# Shallow network
time1 = time.time()
Shallow_dict = {}
train_error_10 = np.zeros((10, epochs))
test_error_10 = np.zeros(10)
test_accuracy_10 = np.zeros(10)
for r in range(10):
    print("Shallow, Repeat: {}.".format(r))
    SNN = LrnetShallow(N_in, N_out)
    train_error, test_error, test_accuracy = run_trial(SNN, N_in, N_hidden, N_out,\
                       train_data, b_size, b_num, test_data, epochs, gamma, lr)
    train_error_10[r] = train_error*1
    test_error_10[r] = test_error*1
    test_accuracy_10[r] = test_accuracy*1
Shallow_dict['train_error'] = train_error_10*1
Shallow_dict['test_error'] = test_error_10*1
Shallow_dict['test_accuracy'] = test_accuracy_10*1
np.save("Shallow_dict.npy", Shallow_dict)
time2 = time.time()
print("Total time cost: {}".format(time2-time1))