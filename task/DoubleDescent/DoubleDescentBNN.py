# =============================================================================
# Double Descent test for Lrnet10
# =============================================================================

#%% import module
import numpy as np
import time
from model.NeuralNetwork.Lrnet import Lrnet
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("N_in", type=int)
# args = parser.parse_args()

# def print_log(content):
#     log = open("logLrnet10_{}.txt".format(args.N_in),"a")
#     print(content, file=log)
#     log.close()  

#%% function
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

def run_trial(N_in, N_hidden, N_out, train_data, b_size, b_num, test_data, \
              epochs, gamma, lr_initial):
    NN = Lrnet(N_in, N_hidden, N_out) 
    train_error_traj = np.zeros(int(epochs/100))
    for i in range(epochs):
        lr = lr_initial*(0.1**((i+1)//25000))
        train_error = train(NN, b_size, b_num, train_data, gamma, lr)
        if (i+1)%100 == 0:
            train_error_traj[int((i+1)/100)-1] = train_error
        print("\rTrainig Progress: {:.2f}%".format((i+1)/epochs*100), end='')
    print("")
    [test_error, test_accuracy] = np.array([test(NN, test_data) for j in range(0,20)]).mean(0)  
    return train_error_traj, test_error, test_accuracy


#%% load data
data_path = "D:/PMI/Projects/LLDL/task/DoubleDescent/PCA"
train_data_all = np.load(data_path + "/trainPCA50.npy")
test_data_all = np.load(data_path + "/testPCA50.npy")


#%%% hyper-parameter
N_in = 50
N_out = 10
p = 2000
b_size = p
b_num = 1 
lr = 0.5
gamma = 1e-5
epochs = 100000

train_data = train_data_all[0:p,0:(1+N_in)]
test_data = test_data_all[:,0:(1+N_in)]
print("Training N_in: {}.".format(N_in))

#%% main
time1 = time.time()
DoubleDescent_dict = {}
for N_hidden in range(1,51,1):
    print("Hidden Nodes: {},".format(N_hidden))
    DoubleDescent_dict[N_hidden] = {}
    train_error, test_error, test_accuracy = run_trial(N_in, N_hidden, N_out,\
                       train_data, b_size, b_num, test_data, epochs, gamma, lr)
    DoubleDescent_dict[N_hidden]['train_error'] = train_error*1
    DoubleDescent_dict[N_hidden]['test_error'] = test_error*1
    DoubleDescent_dict[N_hidden]['test_accuracy'] = test_accuracy*1
np.save("DD_dict_Lrnet10_{}.npy".format(N_in), DoubleDescent_dict)
time2 = time.time()
print("Total time cost: {}".format(time2-time1))