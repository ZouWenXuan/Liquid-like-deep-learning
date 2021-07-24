# =============================================================================
# Generate sigma by Monte Carlo
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
import numpy as np
from model.Tools.MonteCarlo import MonteCarlo  
from model.NeuralNetwork.Lrnet import Lrnet
from model.NeuralNetwork.LrnetShallow import LrnetShallow

#%% Monte Carlo run
path = "./ModelParameter/origin/BetheNew/under_para"
J = np.loadtxt(path + "/J_FULL.txt")
h = np.loadtxt(path + "/h_FULL.txt")
sigma_list = np.loadtxt(path + "/data_100.txt")[0:20,1:]
print(path)

# test data
data_path = "D:/PMI/Projects/LLDL/data/PCA_Mnist"
data_all = np.loadtxt(data_path+"/test_data.txt")

#%%% Monte Carlo function
def MonteCarlo_Modelavg(J, h, beta, relaxation, interval, M):
    monte_carlo_sampling = MonteCarlo(beta, relaxation, interval, M)
    sigma = monte_carlo_sampling.Sampling(J, h)
    mi_model = np.array(sigma.mean(0), ndmin=2).T
    Ca_temp = 0
    for i in range(M):
        Ca_temp += np.dot( (sigma[i:(i+1)]).T, sigma[i:(i+1)])
    Ca_model = Ca_temp/M
    Ca_model[np.diag_indices_from(Ca_model)] = 0
    MonteCarlo_dict = {}
    MonteCarlo_dict['sigma'] = sigma*1
    MonteCarlo_dict['mi'] = mi_model*1
    MonteCarlo_dict['Ca'] = Ca_model*1
    return MonteCarlo_dict

def MarkovStep(J, h, sigma, beta, algorithm='Metropolis-Hastings'):  
    N = J.shape[0]
    # random choose a spin
    spin = np.random.permutation(N)[0] 
    H = np.dot(J[spin:spin+1], sigma) + h[spin]
    if algorithm == 'Metropolis-Hastings':
        Delta_E = 2*H*sigma[spin]
        Trans = np.minimum(1, np.exp(-beta*Delta_E))
    elif algorithm == 'Heat-bath':
        Trans = 1/2*(1-sigma[spin]*np.tanh(beta*H))
    else:
        raise ValueError("Choose Metropolis-Hastings or Heat-bath!")  
    unirand = np.random.uniform(low=0, high=1)          
    Trans_bool = np.sign(unirand-Trans)
    sigma[spin] = sigma[spin]*Trans_bool
    E = -0.5*np.dot(sigma.reshape(1,-1), np.dot(J, sigma.reshape(-1,1)))\
        - np.dot(h.reshape(1,-1), sigma.reshape(-1,1))
    return sigma*1, E[0][0]

def MonteCarlo_SetInitial(J, h, sigma, beta, M):
    sigmas = []
    Es = []
    for i in range(M):
        sigma, E = MarkovStep(J, h, sigma*1, beta)
        sigmas.append(1*sigma.reshape(-1))
        Es.append(E)
        print('\rMonte Carlo Progress: {:.2f}%'.format(len(sigmas)/M*100), end='')
    print('')
    return np.array(sigmas), np.array(Es)


#%%% Accuracy function
def Accuracy(sigma, data, DNN, N_in, N_hid, N_out):
    # segment and assign   
    w_hi = sigma[0:300].reshape(N_hid,N_in)
    w_hh = sigma[300:525].reshape(N_hid,N_hid)
    w_oh = sigma[525:675].reshape(N_out,N_hid)
    DNN.w_hi = w_hi
    DNN.w_hh = w_hh
    DNN.w_oh = w_oh
    # test data
    x = (data[:,1:]).T
    y_test = data[:,0]    
    # forward accuracy
    z_out = DNN.sp_forward(x)
    choice = z_out.argmax(axis=0)
    accuracy = np.mean(y_test == choice)   
    return accuracy

def AccuracyShallow(sigma, data, Shallow, N_in, N_out):
    # assign   
    w = sigma.reshape(N_out, N_in)
    Shallow.w = w
    # test data
    x = (data[:,1:]).T
    y_test = data[:,0]    
    # forward accuracy
    z_out = Shallow.sp_forward(x)
    choice = z_out.argmax(axis=0)
    accuracy = np.mean(y_test == choice)   
    return accuracy


#%%% Manual test
# MonteCarlo
beta = 1
M = 10
sigma_test = sigma_list[0]
sigmas, Es = MonteCarlo_SetInitial(J, h, sigma_test, beta, int(M*675))

# Accuracy test
N_in = 20
N_hid = 15
N_out = 10
DNN = Lrnet(N_in, N_hid, N_out)  
Shallow = LrnetShallow(N_in, N_out)
accuracy_traj = np.zeros(int(M*675))
for i in range(sigmas.shape[0]):
    # deep
    accuracy_traj[i] = Accuracy(sigmas[i], data_all, DNN, N_in, N_hid, N_out) 
    # shallow
    # accuracy_traj[i] = AccuracyShallow(sigmas[i], data_all, Shallow, N_in, N_out) 

#%%% Formal Test
# Data
data_path = "D:/PMI/Projects/LLDL/data/PCA_Mnist"
data_all = np.loadtxt(data_path+"/test_data.txt")
beta = 1
M = 100

# Network
N_in = 20
N_hid = 15
N_out = 10
DNN = Lrnet(N_in, N_hid, N_out)  
Shallow = LrnetShallow(N_in, N_out)

accuracy_traj = np.zeros((sigma_list.shape[0], int(M*675)))
enengy_traj = np.zeros((sigma_list.shape[0], int(M*675)))
for k,sigma in enumerate(sigma_list):
    sigma_test = sigma_list[k]
    sigmas, Es = MonteCarlo_SetInitial(J, h, sigma_test, beta, int(M*675))    
    enengy_traj[k] = Es*1
    for i in range(sigmas.shape[0]):
        # deep network
        accuracy_traj[k][i] = Accuracy(sigmas[i], data_all, DNN, N_in, N_hid, N_out) 
        # shallow network
        # accuracy_traj[k][i] = AccuracyShallow(sigmas[i], data_all, Shallow, N_in, N_out) 
        print('\rTest sample: {}, Progress: {:.2f}%'\
              .format(k, (i+1)/sigmas.shape[0]*100), end='')
    print('')

#%%% save
MCAE = {}
MCAE['a'] = accuracy_traj*1
MCAE['e'] = enengy_traj*1
np.save("MC_AE_Old.npy", MCAE)

#%%% plot: Deep
a_mean = accuracy_traj.mean(0)
a_std = accuracy_traj.std(0)
E_mean = enengy_traj.mean(0)
E_std = enengy_traj.std(0)
step = range(int(M*675))

import matplotlib.pyplot as plt
plt.figure(figsize=(24,8), dpi=200)
# monte carlo step
# accuracy
ax1 = plt.subplot(121)
step_mc = step[::675]
a_mean_mc = a_mean[::675]
a_std_mc = a_std[::675]
label1, = ax1.plot(step_mc, a_mean_mc, color='royalblue', label='Accuracy', lw=4)
plt.fill_between(step_mc, a_mean_mc-a_std_mc, a_mean_mc+a_std_mc,\
                 color='royalblue', alpha=0.5)
plt.ylabel('Accuracy', size=25)
plt.xticks(range(0,67500+6750,675*10), range(0,110,10), size=20)
plt.yticks(size=20)
plt.xlabel('Monte Carlo Epoch', size=25)

# energy
ax2 = ax1.twinx() 
E_mean_mc = E_mean[::675]
E_std_mc = E_std[::675]
label2, = ax2.plot(step_mc, E_mean_mc, color='green', label='Ising Energy',\
                   lw=4, ls='--')
plt.fill_between(step_mc, E_mean_mc-E_std_mc, E_mean_mc+E_std_mc,\
                 color='green', alpha=0.5)

plt.yticks(size=20)
plt.ylabel('Energy', size=25)
plt.title('Monte Carlo trials', size=30)
plt.legend([label1,label2], ['Accuracy','Ising Energy'], loc = 1, fontsize=20) 


ax3 = plt.subplot(122)
step1 = step[0:675]
a_mean1 = a_mean[0:675]
a_std1 = a_std[0:675]
label1, = ax3.plot(step1, a_mean1, color='royalblue', label='Accuracy', lw=4)
plt.fill_between(step1, a_mean1-a_std1, a_mean1+a_std1,\
                 color='royalblue', alpha=0.5)
plt.ylabel('Accuracy', size=25)
plt.xlabel('Monte Carlo step', size=25)
plt.xticks(fontsize=20)
plt.yticks(size=20)

# energy
ax4 = ax3.twinx() 
E_mean1 = E_mean[0:675]
E_std1 = E_std[0:675]
label2, = ax4.plot(step1, E_mean1, color='green', label='Ising Energy',\
                   lw=4, ls='--')
plt.fill_between(step1, E_mean1-E_std1, E_mean1+E_std1, color='green', alpha=0.5)
plt.yticks(size=20)
plt.xlabel('Monte Carlo Epoch', size=25)
plt.ylabel('Energy', size=25)
plt.title('Monte Carlo trials', size=30)
plt.legend([label1,label2], ['Accuracy','Ising Energy'], loc = 1, fontsize=20) 
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylabel('Accuracy', size=25)
plt.title('In one Monte Carlo trial', size=30)
plt.legend(fontsize=20)

plt.subplots_adjust(wspace=0.4)
plt.savefig('MC_AE_Old.png', dpi=200, bbox_inches='tight')

#%%% plot: Shallow
a_mean = accuracy_traj.mean(0)
a_std = accuracy_traj.std(0)
E_mean = enengy_traj.mean(0)
E_std = enengy_traj.std(0)
step = range(int(M*675))

import matplotlib.pyplot as plt
plt.figure(figsize=(24,8), dpi=200)
# monte carlo step
# accuracy
ax1 = plt.subplot(121)
step_mc = step[::675]
a_mean_mc = a_mean[::675]
a_std_mc = a_std[::675]
plt.plot(step_mc, a_mean_mc, color='royalblue', label='Accuracy', lw=4)
plt.fill_between(step_mc, a_mean_mc-a_std_mc, a_mean_mc+a_std_mc,\
                 color='royalblue', alpha=0.5)
plt.ylabel('Accuracy', size=25)
plt.xticks(range(0,67500+6750,675*10), range(0,110,10), size=20)
plt.yticks(size=20)
plt.xlabel('Monte Carlo Epoch', size=25)
plt.legend(fontsize=20)


plt.subplot(122)
# energy
E_mean_mc = E_mean[::675]
E_std_mc = E_std[::675]
plt.plot(step_mc, E_mean_mc, color='green', label='Ising Energy', lw=4, ls='--')
plt.fill_between(step_mc, E_mean_mc-E_std_mc, E_mean_mc+E_std_mc,\
                 color='green', alpha=0.5)
plt.xticks(range(0,67500+6750,675*10), range(0,110,10), size=20)
plt.yticks(size=20)
plt.xlabel('Monte Carlo Epoch', size=25)
plt.ylabel('Energy', size=25)
plt.legend(fontsize=20)

# plt.savefig('MC_AE_Shallow.png', dpi=200, bbox_inches='tight')

#%% Sampling from training
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


#%%% load data
data_path = "D:/PMI/Projects/LLDL/task/DoubleDescent/PCA"
train_data_all = np.load(data_path + "/trainPCA50.npy")
test_data_all = np.load(data_path + "/testPCA50.npy")

#%%%% hyper-parameter
N_in = 20
N_hid = 15
N_out = 10
p = 60000
b_size = 200
b_num = 300 
lr = 0.3
gamma = 1e-5
epochs = 50

train_data = train_data_all[0:p,0:(1+N_in)]
test_data = test_data_all[:,0:(1+N_in)]

#%%%% main
def sampling_from_pw():
    NN = Lrnet(N_in, N_hid, N_out) 
    train_error_traj = np.zeros(epochs)
    for i in range(epochs):
        train_error = train(NN, b_size, b_num, train_data, gamma, lr)
        train_error_traj[i] = train_error
        print("\rTrainig Progress: {:.2f}%".format((i+1)/epochs*100), end='')
    print("")
    a = np.array([test(NN, test_data) for j in range(0,100)])  
    return a[:,1]

accuracy_from_pw = np.zeros((100,100))
for i in range(100):
    print("Sampling repeat: {}".format(i))
    a = sampling_from_pw()
    accuracy_from_pw[i] = a*1


#%%% plot: all samples
plt.figure(figsize=(10,8))
plt.hist(accuracy_from_pw.reshape(-1), bins=50, density=True, edgecolor = 'black', color = 'steelblue')
plt.ylabel('Histogram',size=30)
plt.xlabel(r'Accuracy from $P(W)$',size=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("All Training Samples",size=30)
plt.grid(False)
plt.savefig('TrainingSample_all.png', dpi=200, bbox_inches='tight')

#%%% plot: one sample
plt.figure(figsize=(10,8))
plt.hist(accuracy_from_pw[10], bins=20, density=True, edgecolor = 'black', color = 'steelblue')
plt.ylabel('Histogram',size=30)
plt.xlabel(r'Accuracy from $P(W)$',size=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("One Training Sample",size=30)
plt.grid(False)
plt.savefig('TrainingSample_one.png', dpi=200, bbox_inches='tight')