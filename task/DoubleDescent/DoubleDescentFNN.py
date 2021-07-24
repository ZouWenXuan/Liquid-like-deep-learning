# =============================================================================
# Double Descent test for FNN10
# =============================================================================

#%% import module
import numpy as np
import time
from model.NeuralNetwork.FnnJPA10class import FNN
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("N_in", type=int)
# args = parser.parse_args()

# def print_log(content):
#     log = open("logFNN10_{}.txt".format(args.N_in),"a")
#     print(content, file=log)
#     log.close()  

#%% function
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

def run_trial(N_in, N_hidden, N_out, train_data, test_data, epochs, lr_initial):
    NN = FNN(N_in, N_hidden, N_out) 
    train_error_traj = np.zeros(int(epochs/100))
    for i in range(epochs):
        lr = lr_initial*(0.1**((i+1)//25000))
        train_error = train(NN, train_data, lr)
        if (i+1)%100 == 0:
            train_error_traj[int((i+1)/100)-1] = train_error
            print("\rTrainig Progress: {:.2f}%".format((i+1)/epochs*100), end='')
    test_error, test_accuracy = test(NN, test_data)
    print('')
    return train_error_traj, test_error, test_accuracy


#%% load data
data_path = "D:/PMI/Projects/LLDL/task/DoubleDescent/PCA"
train_data_all = np.load(data_path + "/trainPCA50.npy")
test_data_all = np.load(data_path + "/testPCA50.npy")


#%%% hyper-parameter
N_in = 20
N_out = 10
p = 2000
lr = 1e-2
epochs = 1000000

train_data = train_data_all[0:p,0:(1+N_in)]
test_data = test_data_all[:,0:(1+N_in)]
    
#%% main
time1 = time.time()
DoubleDescent_dict = {}
for N_hidden in range(1,31,1):
    print("Hidden Nodes: {},".format(N_hidden))
    DoubleDescent_dict[N_hidden] = {}
    train_error, test_error, test_accuracy = run_trial(N_in, N_hidden, N_out,\
                                       train_data, test_data, epochs, lr)
    DoubleDescent_dict[N_hidden]['train_error'] = train_error*1
    DoubleDescent_dict[N_hidden]['test_error'] = test_error*1
    DoubleDescent_dict[N_hidden]['test_accuracy'] = test_accuracy*1
    
np.save("DD_dict_FNN10_{}.npy".format(N_in), DoubleDescent_dict)
time2 = time.time()
print("Total time cost: {}".format(time2-time1))