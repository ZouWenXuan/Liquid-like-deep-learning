# =============================================================================
# Permuatation symmetry test: task1
# =============================================================================

#%% import module
import numpy as np
from model.NeuralNetwork.Lrnet import Lrnet

def print_log(content):
    log = open( "log_task1.txt","a")
    print(content, file=log)
    log.close()  

#%% load data
# data (test data)
data_path = "/public/home/huanghp7_zouwx5/LLDL/data/PCA_Mnist/test_data.txt"
data_all = np.loadtxt(data_path)
label = data_all[:,0]
data_one = data_all[np.where(label==1)]

# Parent configuration
accuracy_sigma_all = np.loadtxt("/public/home/huanghp7_zouwx5/LLDL/task/Permutation/new/data_200wPS.txt")

#%% model 
N_in = 20
N_hid = 15
N_out = 10
DNN = Lrnet(N_in, N_hid, N_out)  

#%% permutation function
def permutation_onetime(sigma, data, DNN, N_in, N_hid, N_out, order=False):
    # segment and assign   
    w_hi = sigma[0:300].reshape(N_hid,N_in)
    w_hh = sigma[300:525].reshape(N_hid,N_hid)
    w_oh = sigma[525:675].reshape(N_out,N_hid)
    DNN.w_hi = w_hi
    DNN.w_hh = w_hh
    DNN.w_oh = w_oh
    
    # data
    x = (data[:,1:]).T
    y_test = data[:,0]    
    
    # forward before PS
    z_hid1, z_hid2, z_out = DNN.sp_forward(x, hidden=True)
    choice = z_out.argmax(axis=0)
    accuracy_beforePS = np.mean(y_test == choice)
    
    # PS and forward after PS
    if order:
        PS_hid1 = np.argsort(-z_hid1.mean(1))
        PS_hid2 = np.argsort(-z_hid2.mean(1))
    else:
        PS_hid1 = np.random.permutation(N_hid)
        PS_hid2 = np.random.permutation(N_hid)
    w_hi_ps = w_hi[PS_hid1,:]
    w_hh_ps = w_hh[:,PS_hid1]
    w_hh_ps = w_hh_ps[PS_hid2,:]
    w_oh_ps = w_oh[:,PS_hid2]
    DNN.w_hi = w_hi_ps
    DNN.w_hh = w_hh_ps
    DNN.w_oh = w_oh_ps
    z_hid1_ps, z_hid2_ps, z_out_ps = DNN.sp_forward(x, hidden=True)
    choice_ps = z_out_ps.argmax(axis=0)
    accuracy_afterPS = np.mean(y_test == choice_ps)

    # sigma after PS
    sigma_PS = np.concatenate((w_hi_ps.reshape(-1), w_hh_ps.reshape(-1), w_oh_ps.reshape(-1)))
    return accuracy_beforePS, accuracy_afterPS, sigma_PS


#%% Task
def permutation_1to1(accuracy_sigma, order):
    accuracy_all = (accuracy_sigma[:,0]).reshape(-1,1)   
    sigma_all = accuracy_sigma[:,1:]
    sigma_all_ps = np.zeros(sigma_all.shape)
    M = sigma_all.shape[0]
    for i in range(M):
        a_beforePS, a_afterPS, sigma_PS = permutation_onetime(sigma_all[i],\
                                    data_one, DNN, N_in, N_hid, N_out, order)
        sigma_all_ps[i] = sigma_PS*1
        if (i+1)%10 == 0:    
            print_log("Progress:{:.2f}%, Accuracy test:{}"\
                      .format((i+1)/M*100, a_afterPS==a_beforePS))
    
    # deduplication
    a_ws_ps = np.concatenate((accuracy_all, sigma_all_ps), axis=1)
    print_log("Before unique, shape: {}".format(a_ws_ps.shape))
    a_ws_ps = np.unique(a_ws_ps, axis=0)
    print_log("After unique, shape: {}".format(a_ws_ps.shape))

    # concatenate
    a_ws_all = np.concatenate((accuracy_sigma, a_ws_ps), axis=0)
    return a_ws_all

def permutation_1tomore(accuracy_sigma, Np, order):
    accuracy = accuracy_sigma[0]   
    sigma = accuracy_sigma[1:]
    sigma_all_ps = np.zeros((Np, sigma.size))
    for i in range(Np):
        a_beforePS, a_afterPS, sigma_PS = permutation_onetime(sigma,\
                                    data_one, DNN, N_in, N_hid, N_out, order)
        sigma_all_ps[i] = sigma_PS*1
        if (i+1)%10000 == 0:    
            print_log("Progress:{:.2f}%, Accuracy test:{}"\
                      .format((i+1)/Np*100, a_afterPS==a_beforePS))
    accuracy_all = accuracy*np.ones((Np,1))
    a_ws_ps = np.concatenate((accuracy_all, sigma_all_ps), axis=1)
    return a_ws_ps    

def moments(sigma):
    M,N = sigma.shape   
    # first moment
    mi_data = np.mean(sigma, axis=0)
    print_log("The fisrt moment is computed!")
    
    # second moment
    temp = 0
    for u in range(M):
        temp = temp + np.dot(sigma[u:u+1].T, sigma[u:u+1])    
        if ((u+1)%10000==0):
            print_log("Second Moments, Progress:{:.2f}%".format((u+1)/M*100))
    Ca_data = temp/M
    return mi_data, Ca_data

#%% task 1: ps proportion 
order = False
M = accuracy_sigma_all.shape[0]
for prop in [0.3, 0.6, 0.9]:
    print_log("Test of proportion: {}".format(prop))
    M_ps = int(prop/(2-prop)*M)
    accuracy_sigma_all = np.random.permutation(accuracy_sigma_all)
    a_sigma_ps = accuracy_sigma_all[0:M_ps]
    a_sigma_nps = accuracy_sigma_all[M_ps:]
    # ps part after ps
    a_sigma_ps = permutation_1to1(a_sigma_ps, order)
    accuracy_sigma_all = np.concatenate((a_sigma_ps, a_sigma_nps), axis=0)
    mi_data, Ca_data = moments(1*accuracy_sigma_all[:,1:])  
    np.save('data_PST1_{:.1f}.npy'.format(prop), accuracy_sigma_all)
    np.save('mi_data_PST1_{:.1f}.npy'.format(prop), mi_data)    
    np.save('Ca_data_PST1_{:.1f}.npy'.format(prop), Ca_data)